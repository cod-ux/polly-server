import toml
import os
from openai import OpenAI
import pandas as pd
from datasets import Dataset

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from phoenix.otel import register

from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
    LLMContextRecall,
)
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


import streamlit as st

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

chroma_dir = os.path.join(BASE_DIR, "..", "chroma_db")

# Prepare vector store
vector_store = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings,
    collection_name="polly-rag",
)

os.environ["OPENAI_API_KEY"] = API_KEY

# Prepare LLM Judge
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def query_db(query: str):
    results = vector_store.similarity_search(query=query, k=3)

    return results


# Prepare chat function
def generate_response(question):
    # Query VectorDB
    source = query_db(question)
    context = [page.page_content for page in source]
    string_context = "\n".join(context)

    with open(os.path.join(prompt_folder, "system.md"), "r", encoding="utf-8") as file:
        system_prompt = file.read()

    with open(os.path.join(prompt_folder, "user.md"), "r", encoding="utf-8") as file:
        user_prompt = file.read()

    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(context=string_context),
                },
                {"role": "user", "content": user_prompt.format(question=question)},
            ],
        )
        .choices[0]
        .message.content
    )

    return response, context


# Evaluation
# Prepare dataset - List of dicts [{"reference": "", "query": "", "response": ""}...]
# Context precision, Context recall, Response Relevancy, Faithfullness

# Iterate through queries - make dataframe with reference, query, response


def build_eval_ds(queries_df, evals_df):
    print("Building evaluation dataset...")
    questions = []
    answers = []
    contexts = []
    ground_truth = []
    for index, row in queries_df.head(3).iterrows():
        length = len(queries_df.head(3))
        print(f"Working on question: {index+1}/{length}")
        questions.append(row["Queries"])
        ground_truth.append(row["Ground truth"])
        answer, context = generate_response(row["Queries"])
        answers.append(answer)
        contexts.append(context)

    evals_df["user_input"] = questions
    evals_df["response"] = answers
    evals_df["retrieved_contexts"] = contexts
    evals_df["reference"] = [str(truth) for truth in ground_truth]

    evals_dataset = EvaluationDataset.from_pandas(evals_df)
    return evals_dataset


def run_eval(df, llm=evaluator_llm, emd=evaluator_embeddings):
    print("Running evaluation...")
    metrics = [
        LLMContextRecall(llm=llm),
        FactualCorrectness(llm=llm),
        Faithfulness(llm=llm),
        SemanticSimilarity(embeddings=evaluator_embeddings),
    ]
    results = evaluate(dataset=df, metrics=metrics).to_pandas()

    return results


eval_df = pd.DataFrame()
query_df = pd.read_excel(queries_path)

# Initating test

response_ds = build_eval_ds(queries_df=query_df, evals_df=eval_df)
eval_results = run_eval(response_ds)

filename = input("Save table as: ")

Saved = False
while not Saved:
    if filename:
        print("Filename exists...")
        if not os.path.exists(
            os.path.join(output_folder, "output", f"{filename}.xlsx")
        ):
            print("Filename is valid...")
            eval_results.to_excel(
                os.path.join(output_folder, "output", f"{filename}.xlsx"),
                index=False,
            )
            print("Saved file...")
            print(f"Save {filename}.xlsx successfully")
            Saved = True

        else:
            print("File already exists")

    else:
        print("Please enter a valid name to save the file.")
