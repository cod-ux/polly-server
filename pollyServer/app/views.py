# chatapp/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import OpenAI
from django.conf import settings
import os

from app.RAG.rag import query_db, query_model

BASE_DIR = os.path.dirname(__file__)

prompt_folder = os.path.join(BASE_DIR, "RAG", "prompts")

with open(os.path.join(prompt_folder, "system.md"), "r", encoding="utf-8") as file:
    system_prompt = file.read()

with open(os.path.join(prompt_folder, "user.md"), "r", encoding="utf-8") as file:
    user_prompt = file.read()


@csrf_exempt  # Enabled Cross Origin for development
def chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Check if messages array is present and valid
            if not isinstance(data, list) or len(data) > 5 or len(data) == 0:
                return JsonResponse({"error": "Invalid input: Expected array of 1-5 messages"}, status=400)

            # Validate message format
            for msg in data:
                if not isinstance(msg, dict) or "sender" not in msg or "msg" not in msg:
                    return JsonResponse({"error": "Invalid message format"}, status=400)
                if msg["sender"] not in ["user", "bot"]:
                    return JsonResponse({"error": "Invalid sender type"}, status=400)

            # Query VectorDB with the last user message
            last_user_msg = next((msg["msg"] for msg in reversed(data) if msg["sender"] == "user"), None)
            if not last_user_msg:
                return JsonResponse({"error": "No user message found"}, status=400)

            response = query_model(last_user_msg, data)
            return JsonResponse({"response": response}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
