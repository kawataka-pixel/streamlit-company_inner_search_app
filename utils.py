"""
このファイルは、画面表示以外の処理用関数をまとめたファイルです。
"""

from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

import constants as ct

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_llm_response(chat_message: str):
    """
    LangChain 1.x 構成に合わせた処理で回答を生成する。
    ステップ: 履歴を独立化 → 検索 → 回答生成。
    戻り値: {"answer": str, "context": List[Document]}
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    retriever = st.session_state.get("retriever")
    if retriever is None:
        st.error("🔎 検索エンジン（retriever）が未初期化です。initialize.py で設定してください。")
        return {"answer": "現在、検索準備中です。しばらく待ってから再試行してください。", "context": []}

    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    reform_prompt = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    refine_chain = reform_prompt | llm | StrOutputParser()

    refined_query = refine_chain.invoke({
        "input": chat_message,
        "chat_history": st.session_state.chat_history,
    })

    docs_for_answer = retriever.invoke(refined_query)
    # LangChainの変換でクエリが変わりすぎてヒットしない場合のフォールバック
    if not docs_for_answer:
        docs_for_answer = retriever.invoke(chat_message)

    context_text = "\n\n".join(doc.page_content for doc in docs_for_answer)

    qa_system = ct.SYSTEM_PROMPT_DOC_SEARCH if st.session_state.mode == ct.ANSWER_MODE_1 else ct.SYSTEM_PROMPT_INQUIRY
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system),
        MessagesPlaceholder("chat_history"),
        ("human", "質問: {input}\n\n参照文書:\n{context}"),
    ])
    answer_runnable = qa_prompt | llm | StrOutputParser()
    answer = answer_runnable.invoke({
        "input": chat_message,
        "chat_history": st.session_state.chat_history,
        "context": context_text,
    })

    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message),
        AIMessage(content=answer),
    ])

    return {"answer": answer, "context": docs_for_answer}


def build_error_message(msg: str) -> str:
    """エラー表示用のメッセージを整形"""
    return f"⚠️ {msg}\n\n対処: 設定(.env/Secrets)やAPIキー、依存ライブラリ、初期化処理を確認して再実行してください。"



def get_source_icon(source: str) -> str:
    """
    参照元の文字列から表示用のアイコンを返す。
    - URL(http/https)なら LINK_ICON
    - それ以外（ローカル/パス/空）は DOC_ICON or WARNING_ICON
    """
    if not source or not str(source).strip():
        return ct.WARNING_ICON

    source_str = str(source).strip().lower()
    if source_str.startswith('http://') or source_str.startswith('https://'):
        return ct.LINK_SOURCE_ICON

    return ct.DOC_SOURCE_ICON


def build_excerpt(text: str, max_chars: int = 160) -> str:
    """テキストから簡易的な抜粋を生成する"""
    if not text:
        return ""

    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact

    return compact[:max_chars].rstrip() + "..."
