# chatapp/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import OpenAI
from django.conf import settings
import os

from app.RAG.rag import query_db

BASE_DIR = os.path.dirname(__file__)

prompt_folder = os.path.join(BASE_DIR, "RAG", "prompts")

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

with open(os.path.join(prompt_folder, "system.md"), "r", encoding="utf-8") as file:
    system_prompt = file.read()

with open(os.path.join(prompt_folder, "user.md"), "r", encoding="utf-8") as file:
    user_prompt = file.read()


@csrf_exempt  # Enabled Cross Origin for development
def chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Check if the "message" field is present
            if "message" not in data:
                return JsonResponse({"error": "Invalid input"}, status=400)

            # Query VectorDB
            response = query_db(data["message"])

            context = "\n".join([page.page_content for page in response])
            # Call OpenAI API

            response = (
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt.format(context=context),
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(question=data["message"]),
                        },
                    ],
                )
                .choices[0]
                .message.content
            )

            return JsonResponse({"response": response}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
