"""
このファイルは、画面表示以外の汎用処理をまとめたユーティリティです。
"""

from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

import constants as ct


# .env に定義された環境変数を読み込み
load_dotenv()


def get_source_icon(source: str) -> str:
    """
    参照元（ファイル / URL）に応じた表示用アイコンを返す。
    """
    if not source or not str(source).strip():
        return ct.WARNING_ICON

    src = str(source).strip().lower()
    if src.startswith("http://") or src.startswith("https://"):
        return ct.LINK_SOURCE_ICON
    return ct.DOC_SOURCE_ICON


def build_error_message(message: str) -> str:
    """
    エラーメッセージと共通テンプレートを連結して返す。
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def _ensure_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _build_independent_query(llm: ChatOpenAI, chat_message: str) -> str:
    """
    会話履歴なしでも理解できる独立したクエリを生成する。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "input": chat_message,
            "chat_history": st.session_state.chat_history,
        }
    )


def _generate_answer(llm: ChatOpenAI, chat_message: str, context_text: str) -> str:
    """
    RAGで取得した文脈 + 会話履歴をもとに回答を生成する。
    """
    qa_system = (
        ct.SYSTEM_PROMPT_DOC_SEARCH
        if st.session_state.mode == ct.ANSWER_MODE_1
        else ct.SYSTEM_PROMPT_INQUIRY
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system),
            MessagesPlaceholder("chat_history"),
            ("human", "質問: {input}\n\n参照文書:\n{context}"),
        ]
    )
    answer_chain = qa_prompt | llm | StrOutputParser()
    return answer_chain.invoke(
        {
            "input": chat_message,
            "chat_history": st.session_state.chat_history,
            "context": context_text,
        }
    )


def get_llm_response(chat_message: str):
    """
    LangChain 1.x の Runnable API を用いて RAG 応答を生成する。

    Returns:
        {"answer": str, "context": List[Document]}
    """
    _ensure_chat_history()

    retriever = st.session_state.get("retriever")
    if retriever is None:
        st.error("🔎 検索エンジン（retriever）が未初期化です。initialize.py で設定してください。")
        return {
            "answer": "現在、検索準備中です。しばらく待ってから再試行してください。",
            "context": [],
        }

    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    refined_query = _build_independent_query(llm, chat_message)

    docs_for_answer = retriever.invoke(refined_query)
    if not docs_for_answer:
        # 質問の変形でヒットしない場合は元の入力で再検索
        docs_for_answer = retriever.invoke(chat_message)

    context_text = "\n\n".join(doc.page_content for doc in docs_for_answer)

    answer = _generate_answer(llm, chat_message, context_text)

    st.session_state.chat_history.extend(
        [
            HumanMessage(content=chat_message),
            AIMessage(content=answer),
        ]
    )

    return {"answer": answer, "context": docs_for_answer}
