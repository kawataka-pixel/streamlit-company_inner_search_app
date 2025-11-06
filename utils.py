"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st

# ==========================================================
# LangChain（最新版0.3系）対応の新しいインポート構成
# ==========================================================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import constants as ct

############################################################
# 設定関連
############################################################
load_dotenv()

# （任意）起動時に履歴だけ初期化しておくと安心
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

############################################################
# 関数定義
############################################################
def get_llm_response(chat_message: str):
    """
    LangChain 1.x 構成（Runnable）版。
    履歴を見て質問を独立化 → retriever検索 → 回答生成。
    返り値: {"answer": str, "context": List[Document]}
    """
    # --- 安全ガード：ここは必ず関数内で！ ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "retriever" not in st.session_state or st.session_state.retriever is None:
        st.error("🔎 検索エンジン（retriever）が未初期化です。initialize.py で設定してください。")
        return {"answer": "現在、検索準備中です。しばらく待ってから再試行してください。", "context": []}

    # --- LLM ---
    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    # --- ① 履歴を見て質問を独立化（旧 create_history_aware_retriever 相当） ---
    reform_prompt = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    refine_chain = reform_prompt | llm | StrOutputParser()   # -> refined_query(str)

    # --- ② 検索（refined_query を retriever に渡す） ---
    def _retrieve(inputs):
        q = inputs["refined_query"]
        return st.session_state.retriever.get_relevant_documents(q)

    # --- ③ 回答プロンプト（モード別） ---
    qa_system = ct.SYSTEM_PROMPT_DOC_SEARCH if st.session_state.mode == ct.ANSWER_MODE_1 else ct.SYSTEM_PROMPT_INQUIRY
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system),
        MessagesPlaceholder("chat_history"),
        ("human", "質問: {input}\n\n参考文書:\n{context}")
    ])

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # --- Runnableグラフ（旧 create_retrieval_chain / create_stuff_documents_chain の代替） ---
    chain = (
        {
            "chat_history": RunnablePassthrough(),
            "input": RunnablePassthrough(),
            "refined_query": refine_chain,
        }
        | {
            "chat_history": lambda x: x["chat_history"],
            "input":        lambda x: x["input"],
            "context":      lambda x: _format_docs(_retrieve(x)),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "input": chat_message,
        "chat_history": st.session_state.chat_history
    })

    # 履歴に追加
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message),
        AIMessage(content=answer),
    ])

    # ソース表示用（不要なら削除OK）
    docs = st.session_state.retriever.get_relevant_documents(chat_message)
    return {"answer": answer, "context": docs}
