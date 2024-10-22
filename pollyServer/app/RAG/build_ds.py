import os
import toml
from openai import OpenAI
import random

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source_folder_path = os.path.join(BASE_DIR, "docs")
results_path = os.path.join(BASE_DIR, "eval-ds", "qas.txt")
secrets_path = os.path.join(BASE_DIR, "..", "config", "secrets.toml")

API_KEY = toml.load(secrets_path)["OPENAI_API_KEY"]


# print(os.path.exists(secrets_path)) Validate path

"""
Build dataset of questions, draft answers, source pages.
Note: Need to build a dataset of 20 Q&A's that have a mix of recommendations

Logic:
1. Iterate through the file_path in source_folder_path
2. Turn pdf into chroma dbs
2. If chroma_dbs is "close enough to care" get 3 percentage related questions
3. If chroma_dbs is "making the grade" get 3 survey data related questions
4. If chroma_dbs is other 5 papers generate 2, 4x3 questions each
5. Add questions & answers at each step to qas.txt
"""

client = OpenAI(api_key=API_KEY)
embedding_model = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model=embedding_model, api_key=API_KEY)


qa_prompt_1 = """
Your task is to write an important factoid question and an answer for researchers given a context.
Your Factoid question should be answerable with a specific, concise piece of factual information from the context relating to the PER CENT information given.
Your Factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid Question: (your question)
Answer: (your answer to the question)


Now here is the context.

Context: {context}\n
Output:::
"""  # Close enough to care - 3 percent QA

qa_prompt_2 = """
Your task is to write an important factoid question and an answer for researchers given a context.
Your question should be answerable with a specific, concise piece of factual information from the context relating to SURVEY DATA information given.
Your question should be formulated in the same style as questions users could ask in a search engine.
This means that your question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid Question: (your question)
Answer: (your answer to the question)

Now here is the context.

Context: {context}\n
Output:::
"""  # Making the grade - 3 survey QA
qa_prompt_3 = """
Your task is to write an important factoid question and an answer for researchers given a context.
Your question should be answerable with a piece of relevant information from the context.
Your question should be formulated in the same style as questions users could ask in a search engine.
Your answer should be rich with factual information with deep explanations, yet concise.

Provide your answer as follows:

Output:::
Factoid question: (your question)
Answer: (your answer to the question)

Now here is the context.

Context: {context}\n
Output:::
"""  # Rest 5 papers - 2,3,3,3,3 QA


def generate_qa(search_aim, prompt, dbs: Chroma):

    results = dbs.similarity_search(query=search_aim, k=12)

    page_count = random.randint(0, 11)

    context = results[page_count].page_content

    response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(context=context),
                },
            ],
            temperature=0.8,
        )
        .choices[0]
        .message.content
    )

    with open(results_path, "a+") as file:
        file.write(
            "\n"
            + response
            + "\n"
            + f"Source: {os.path.basename(results[0].metadata['source'])}"
            + "\n"
            + f"Page: {results[page_count].metadata['page']}"
            + "\n"
            + "---"
            + "\n"
        )


# Function to create document objects from file path including page number & book title
def make_dbs(pdf_path):
    pages = []
    if pdf_path[-4:] == ".pdf":
        print("Name of pdf: ", pdf_path)
        loader = PyPDFLoader(os.path.join(BASE_DIR, "docs", pdf_path))

        for page in loader.lazy_load():
            pages.append(page)

    else:
        pass

    print("No. of source pages: ", len(pages))

    chroma_dbs = Chroma.from_documents(
        documents=pages,
        embedding=embeddings,
    )

    return chroma_dbs


for index, file_path in enumerate(os.listdir(source_folder_path)):
    print(f"File index: {index}")
    if file_path.endswith(".pdf"):
        dbs = make_dbs(file_path)
        if file_path == "Close-enough-to-care.pdf":
            for i in range(3):
                generate_qa(prompt=qa_prompt_1, search_aim="Percentage facts", dbs=dbs)

        elif file_path == "Making-the-grade.pdf":
            for i in range(3):
                generate_qa(prompt=qa_prompt_2, search_aim="Survey data", dbs=dbs)

        elif file_path == "The-power-of-prevention-3.pdf":
            for i in range(2):
                generate_qa(prompt=qa_prompt_3, search_aim="", dbs=dbs)

        else:
            for i in range(3):
                generate_qa(prompt=qa_prompt_3, search_aim="", dbs=dbs)
