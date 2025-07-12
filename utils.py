from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dashscope import TextEmbedding
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os

class TongyiEmbeddings(Embeddings):
    def __init__(self):
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

    def embed_documents(self, texts):
        return [TextEmbedding.call(input=text, model="text-embedding-v1").output["embeddings"][0]["embedding"] for text in texts]

    def embed_query(self, text):
        return TextEmbedding.call(input=text, model="text-embedding-v1").output["embeddings"][0]["embedding"]

def qa_agent(qwen_api_key, memory, question):
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=qwen_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    loader = PyPDFLoader("teacher_message.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = TongyiEmbeddings()
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    prompt_template = """
    你是一个智能助手，根据给定的文档内容回答问题。
    如果没有找到与问题相关的信息，请回复：请重新输入与老师信息有关的问题。

    {context}

    问题：{question}
    """
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    chat_history = memory.load_memory_variables({})["chat_history"]
    response = qa.invoke({"chat_history": chat_history, "question": question})
    return response