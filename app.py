# import Essential dependencies
import streamlit as sl
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama



# function to load the vectordatabase
def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="llama3")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return db
# function to load the LLM
def load_llm():
    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo", temperature=0, api_key="Enter your api key"
    # )
    llm = Ollama(model="llama3")
    return llm

# creating prompt template using langchain
def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf answer "i donot know what the hell you are asking about"
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
        similar_embeddings = knowledgeBase.similarity_search(query)
        similar_embeddings = FAISS.from_documents(
            documents=similar_embeddings,
            # embedding=OpenAIEmbeddings(api_key="Enter your api key"),
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
