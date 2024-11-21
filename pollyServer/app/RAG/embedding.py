# Build vector db for pdf(s)

import toml
import os
from openai import OpenAI
import asyncio
from docx import Document
import docx2txt2 as docx2txt

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        if pdf_path[-4:] == ".pdf":
            print("Name of pdf: ", pdf_path)
            loader = PyPDFLoader(os.path.join(BASE_DIR, "docs", pdf_path))

            async for page in loader.alazy_load():
                pages.append(page)

        else:
            pass

    print("No. of source pages: ", len(pages))


asyncio.run(load_pdfs())

# Create source

source = []

for page in pages:
    og_content = page.page_content
    book_name = os.path.basename(page.meta_data.source)

    new_content = "Source name: " + book_name + "\n\n" + og_content

    source.append(new_content)

# Create source docx
print("Loading Document class")
document = Document()

for page in pages:
    print(f"Adding page: {page.metadata}")
    document.add_paragraph(page.page_content)

print("Iteration complete")
document_path = os.path.join(BASE_DIR, "docs", "source.docx")
document.save(document_path)

# Text splitting method
source = docx2txt.extract_text(document_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.create_documents([source])
print(f"Length of chuks: {len(chunks)}")

# Create a Vectorstore
chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=os.path.join(BASE_DIR, "chroma_db"),
    collection_name="polly-rag",
)
