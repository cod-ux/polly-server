# chatapp/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import OpenAI
from django.conf import settings

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

            # Call OpenAI API
            response = (
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": data["message"],
                        }
                    ],
                )
                .choices[0]
                .message.content
            )

            return JsonResponse({"response": response}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
