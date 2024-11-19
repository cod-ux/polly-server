# Build vector db for pdf(s)

import toml
import os
from openai import OpenAI
import asyncio
from docx import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # path of the current folder that holds this file

# secrets_path = os.path.join(BASE_DIR, "..", "config", "secrets.toml")
folder_path = os.path.join(BASE_DIR, "..", "docs")

# API_KEY = toml.load(secrets_path)["OPENAI_API_KEY"]

# embedding_model = "text-embedding-3-large"
# embeddings = OpenAIEmbeddings(model=embedding_model, api_key=API_KEY)

# client = OpenAI(api_key=API_KEY)
pages = []


async def load_pdfs():
    for pdf_path in os.listdir(folder_path):
        if pdf_path[-4:] == ".pdf":
            print("Name of pdf: ", pdf_path)
            loader = PyPDFLoader(os.path.join(BASE_DIR, "..", "docs", pdf_path))

            async for page in loader.alazy_load():
                pages.append(page)

        else:
            pass

    print("No. of source pages: ", len(pages))


asyncio.run(load_pdfs())

# Create a word document of the pages
print("Loading Document class")
document = Document()

for page in pages:
    print(f"Adding page: {page.metadata}")
    document.add_paragraph(page.page_content)

print("Iteration complete")
document.save(os.path.join(BASE_DIR, "..", "docs", "source.docx"))


# Splitting method


# Creating vector DB
