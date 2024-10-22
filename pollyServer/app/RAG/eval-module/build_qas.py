import os
import toml
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(BASE_DIR, "..", "..", "config", "secrets.toml")
source_docs_folder_path = os.path.join(BASE_DIR, "..", "docs")

# print(os.path.exists(secrets_path)) Validate path

"""
Build dataset of questions, draft answers, source pages.

Note: Need to build a dataset of 20 Q&A's that have a mix of recommendations
"""
