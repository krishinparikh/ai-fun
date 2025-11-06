from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import openai
import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Parse and load document
# Split document into chunks
# Store chunks in vector database


DATA_PATH = "./DesignPatterns.pdf"
CHROMA_PATH = "chroma"

template = """
Answer the question based only on the following text:
{context}

Answer the question based on the context above: {question}
"""

def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def retrieve_context(question: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search(question, k=3)
    return results