# import Essential dependencies

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# create a new file named vectorstore in your current directory.
if __name__ == "__main__":
    DB_FAISS_PATH = "vectorstore/db_faiss"
    '''#loader = PyPDFLoader("/Users/megharoshan/vscode_projects/Inventory-retrieval-through-LLM-and-RAG/1706.03762v7.pdf")
    loader = CSVLoader("/Users/megharoshan/vscode_projects/Inventory-retrieval-through-LLM-and-RAG/data.csv")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(splits)
    vectorstore = FAISS.from_documents(
        documents=splits, embedding=OllamaEmbeddings(model="llama3")
    )
    vectorstore.save_local(DB_FAISS_PATH)'''
    loader = CSVLoader(
        "/Users/megharoshan/vscode_projects/Inventory-retrieval-through-LLM-and-RAG/data.csv",
        csv_args={"delimiter": ","},
        source_column="Description",
        encoding="unicode_escape",
    )
    docs = loader.load()[:100]
    print("docs")
    print(docs[:3])
    print("#" * 50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        documents=splits, embedding=OllamaEmbeddings(model="llama3")
    )
    vectorstore.save_local(DB_FAISS_PATH)