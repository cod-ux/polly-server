# Build vector db for pdf(s)

import toml
import os
from openai import OpenAI
import asyncio

from langchain_community.document_loaders import PyPDFLoader
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
pages = []


async def load_pdfs():
    for pdf_path in os.listdir(folder_path):
        loader = PyPDFLoader(os.path.join(BASE_DIR, "docs", pdf_path))

        async for page in loader.alazy_load():
            pages.append(page)

    print("No. of source pages: ", len(pages))


asyncio.run(load_pdfs())

# Create a Vectorstore
chroma_db = Chroma.from_documents(
    documents=pages,
    embedding=embeddings,
    persist_directory=os.path.join(BASE_DIR, "chroma_db"),
    collection_name="polly-rag",
)
