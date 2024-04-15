import weaviate
import json
import requests
import tika
from tika import parser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma
from dotenv import load_dotenv
import os

load_dotenv()
chat_gpt_api_key = os.getenv("CHAT_GPT_API_KEY")

client = weaviate.Client(
    url = "http://localhost:8080",  # Replace with your endpoint
    #auth_client_secret=weaviate.auth.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-OpenAI-Api-Key": chat_gpt_api_key  # Replace with your inference API key
    }
)

class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-openai": {},
        "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
    }
}

# client.schema.create_class(class_obj)

# resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
# data = json.loads(resp.text)  # Load data

text = parser.from_file("./dataset.pdf")
text_splitter = CharacterTextSplitter(chunk_size=8191, chunk_overlap=200)
data = text_splitter.create_documents([text["content"]])

client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:  # Initialize a batch process
    for i, d in enumerate(data):  # Batch import data
        print(f"importing question: {i+1}")
        # print(d.page_content)
        properties = {
            "content": d.page_content
        }
        batch.add_data_object(
            data_object=properties,
            class_name="Question"
        )

print(f"process completed")