import toml
import os
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # path of the current folder that holds this file


secrets_path = os.path.join(BASE_DIR, "..", "config", "secrets.toml")
folder_path = os.path.join(BASE_DIR, "docs")

API_KEY = toml.load(secrets_path)["OPENAI_API_KEY"]

embedding_model = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model=embedding_model, api_key=API_KEY)

client = OpenAI(api_key=API_KEY)

chroma_dir = os.path.join(BASE_DIR, "chroma_db")

vector_store = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings,
    collection_name="polly-rag",
)


def query_db(query: str):
    results = vector_store.similarity_search(query=query, k=3)

    return results
