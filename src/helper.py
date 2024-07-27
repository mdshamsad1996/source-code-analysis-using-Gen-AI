import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# clone any github repositories
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url,to_path=repo_path)

# Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob='**/*',
                                           suffixes=[".py"],
                                           parser=LanguageParser(language=Language.PYTHON,parser_threshold=500)
                                           )

    dcouments = loader.load()
    return dcouments


# creating text chunk
def text_splitter(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=200)
    text_chunks = document_splitter.split_documents(documents)
    return text_chunks


# Load Embedding models
def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings
