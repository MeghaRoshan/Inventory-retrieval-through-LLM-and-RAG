# import Essential dependencies
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["HF_HOME"] = "/Users/megharoshan/vscode_projects/Inventory-retrieval-through-LLM-and-RAG/hf-model/"

from abc import ABC
from typing import Any, List, Mapping, Optional

import streamlit as sl
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-0.5B-Instruct",
#     torch_dtype=torch.float16,
#     device_map="cpu",
# )

# # disk_offload(model=model, offload_dir="offload")

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")


# function to load the vectordatabase
def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="llama3")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return db


class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len,
        }


# function to load the OPENAI LLM
def load_llm():
    # from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama

    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo", temperature=0, api_key="Enter your api key"
    # )
    # llm = Qwen()  # Ollama(model="llama3")
    llm = Ollama(model="llama3", temperature=0)

    return llm


# creating prompt template using langchain
def load_prompt():
    # prompt = """ You need to answer the question in the sentence as same as in the  pdf content. .
    #     Given below is the context and question of the user.
    #     context = {context}
    #     question = {question}
    #     if the answer is not in the pdf answer "i donot know what the hell you are asking about"
    #      """

    prompt = """Given the list of products, explain why each product is similar or dissimilar to the query. 
        Given below is the list of products and query.
        list of products = {context}
        query = {question}
        if the query is not in the database "i donot know what the hell you are asking about"
         """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    sl.header("RAG-Langchain")
    sl.write("Enter your queries")
    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()

    query = sl.text_input("Enter some text")

    if query:
        # getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings = knowledgeBase.similarity_search(query, k=10)
        similar_embeddings = FAISS.from_documents(
            documents=similar_embeddings,
            embedding=OllamaEmbeddings(model="llama3"),
        )
        # creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        sl.write(response)
