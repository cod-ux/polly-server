import os
import toml
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(BASE_DIR, "..", "..", "config", "secrets.toml")
source_docs_folder_path = os.path.join(BASE_DIR, "..", "docs")
final_ds_path = os.path.join(BASE_DIR, "..", "eval-ds", "queries.xlsx")

# print(os.path.exists(secrets_path)) Validate path

"""
Build general function for writing & saving QA's with sources

build_qa(folder_path_for_docs) => write QA's to excel file Queries

"""
