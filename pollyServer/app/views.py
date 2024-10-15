# chatapp/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import OpenAI
from django.conf import settings

from app.RAG.rag import query_db

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


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
                            "content": f"Use this Content to respond to the user question. Context:\n\n{context}",
                        },
                        {
                            "role": "user",
                            "content": f"User Question: {data["message"]}\n\nAnswer:\n",
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
