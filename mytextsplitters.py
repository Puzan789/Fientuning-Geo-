import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

class CustomTextSplitter:
    def __init__(self, chunk_size=8000, chunk_overlap=1500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def load_text(self, file_path):
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print(len(documents))
        return documents

    def split_text(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return chunks

        

