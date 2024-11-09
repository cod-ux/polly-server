import toml
import os
from openai import OpenAI
import pandas as pd

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from phoenix.otel import register

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


secrets_path = os.path.join(BASE_DIR, "..", "..", "config", "secrets.toml")
docs_folder = os.path.join(BASE_DIR, "docs")
prompt_folder = os.path.join(BASE_DIR, "prompts")

queries_path = os.path.join(BASE_DIR, "..", "eval-ds", "Queries.xlsx")
output_folder = os.path.join(BASE_DIR, "..", "eval-ds")

API_KEY = toml.load(secrets_path)["OPENAI_API_KEY"]

embedding_model = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model=embedding_model, api_key=API_KEY)

client = OpenAI(api_key=API_KEY)

chroma_dir = os.path.join(BASE_DIR, "chroma_db")

# Prepare vector store
vector_store = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings,
    collection_name="polly-rag",
)


def query_db(query: str):
    results = vector_store.similarity_search(query=query, k=3)

    return results


# Prepare chat function
def chat(question):
    # Query VectorDB
    source = query_db(question)
    context = "\n".join([page.page_content for page in source])

    with open(os.path.join(prompt_folder, "system.md"), "r", encoding="utf-8") as file:
        system_prompt = file.read()

    with open(os.path.join(prompt_folder, "user.md"), "r", encoding="utf-8") as file:
        user_prompt = file.read()

    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": user_prompt.format(question=question)},
            ],
        )
        .choices[0]
        .message.content
    )

    return response


# Evaluation
# Prepare dataset - List of dicts [{"reference": "", "query": "", "response": ""}...]
# Context precision, Context recall, Response Relevancy, Faithfullness

# Iterate through queries - make dataframe with reference, query, response
df = pd.DataFrame()

queries_df = pd.read_excel(queries_path)

for query in queries_df[]
