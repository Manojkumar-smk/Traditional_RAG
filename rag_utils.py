import os
import time
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document

load_dotenv()

def load_and_process_file(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def setup_vectorstore(splits: List[Document]) -> FAISS:
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore

def get_system_prompt() -> str:
    try:
        with open("system_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful assistant."

def get_rag_chain(vectorstore: FAISS):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    retriever = vectorstore.as_retriever()
    system_prompt = get_system_prompt()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("ai", "{context}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def query_rag(rag_chain, input_text: str, session_id: str) -> Tuple[str, float, List[Document]]:
    start_time = time.time()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}},
    )

    end_time = time.time()
    latency = end_time - start_time

    return response["answer"], latency, response["context"]