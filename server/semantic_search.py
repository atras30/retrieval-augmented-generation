import weaviate
import json
from dotenv import load_dotenv
import os

load_dotenv()
chat_gpt_api_key = os.getenv("CHAT_GPT_API_KEY")

client = weaviate.Client(
    url = "http://localhost:8080", 
    additional_headers = {
        "X-OpenAI-Api-Key": chat_gpt_api_key
    }
)

response = (
    client.query
    .get("Question", ["content"])
    .with_near_text({"concepts": ["undang undang pasal 17"]})
    .with_limit(3)
    .do()
)

print(json.dumps(response, indent=4))