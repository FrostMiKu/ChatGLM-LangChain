from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import FAISS
from typing import Optional, List
import os


class ProxyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt


def init_knowledge_vector_store(filepath: str, embeddings):
    md_splitter = MarkdownTextSplitter(chunk_size=256, chunk_overlap=0)
    docs = []
    if not os.path.exists(filepath):
        print("路径不存在")
        return None
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
            print(f"{file} 已成功加载")
        except:
            print(f"{file} 未能成功加载")
            return None
    elif os.path.isdir(filepath):
        for file in os.listdir(filepath):
            fullfilepath = os.path.join(filepath, file)
            try:
                loader = UnstructuredFileLoader(fullfilepath, mode="elements")
                docs += loader.load()
                print(f"{file} 已成功加载")
            except:
                print(f"{file} 未能成功加载")

#    texts = md_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def init_chain_proxy(llm_proxy: LLM, vector_store, top_k=5):
    prompt_template = """你是一个专业的人工智能助手，以下是一些提供给你的已知内容，请你简洁和专业的来回答用户的问题，答案请使用中文。

已知内容:
{context}

参考以上内容请回答如下问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm_proxy,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    return knowledge_chain
