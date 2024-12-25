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

chroma_dir = os.path.join(BASE_DIR, "chroma")
prompt_folder = os.path.join(BASE_DIR, "prompts")

vector_store = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings,
    collection_name="polly-rag",
)


def query_db(query: str):
    results = vector_store.similarity_search(query=query, k=3)

    return results


def query_model(query: str, message_history=None):
    results = vector_store.similarity_search(query=query, k=3)
    context = "\n".join([page.page_content for page in results])

    with open(os.path.join(prompt_folder, "system.md"), "r", encoding="utf-8") as file:
        system_prompt = file.read()

    with open(os.path.join(prompt_folder, "user.md"), "r", encoding="utf-8") as file:
        user_prompt = file.read()

    # Format conversation history
    conversation_history = ""
    if message_history:
        conversation_history = "\n".join([
            f"{'User' if msg['sender'] == 'user' else 'Polly'}: {msg['msg']}"
            for msg in message_history[:-1]  # Exclude the current question
        ])

    # Call OpenAI API
    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt.format(
                        question=query,
                        context=context,
                        conversation_history=conversation_history
                    ),
                },
            ],
        )
        .choices[0]
        .message.content
    )

    return response
