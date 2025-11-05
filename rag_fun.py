from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parse document
# 

DATA_PATH = "./DesignPatterns.pdf"

def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents

