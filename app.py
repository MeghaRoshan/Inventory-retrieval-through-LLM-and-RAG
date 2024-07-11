# import Essential dependencies
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS



# function to load the vectordatabase
def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="llama3")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return db