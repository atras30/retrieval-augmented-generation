import weaviate
import json
import requests
import tika
from tika import parser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma
from dotenv import load_dotenv
from openai import OpenAI
import os
import json


load_dotenv()
chat_gpt_api_key = os.getenv("CHAT_GPT_API_KEY")
gpt_client = OpenAI(api_key=chat_gpt_api_key)
local_url = os.getenv("LOCAL_URL")
weaviate_client = weaviate.Client(
    url = local_url, 
    additional_headers = {
        "X-OpenAI-Api-Key": chat_gpt_api_key
    }
)
weaviate_schema_name = os.getenv("WEAVIATE_SCHEMA_NAME")

WEAVIATE_ROLE = "You are a Vector Database Specialist. Your primary responsibility is to develop and optimize algorithms for parsing user prompts and generating efficient queries for vector databases. Leveraging your expertise in natural language processing (NLP) and database management, you will play a crucial role in enabling seamless interaction between users and the vector database, ensuring accurate and relevant responses to user queries."

CHATBOT_ROLE = "You are a Legal Expert for Indonesian Tax Statutory Rules, your primary responsibility is to provide accurate, reliable, and up-to-date information on Indonesian tax laws and regulations where the data of the laws will be provided along with the prompt. You will serve as a virtual legal expert, assisting users with inquiries, interpretations, and explanations related to tax statutory rules in Indonesia."

WEAVIATE_PROMPT = f"""
        I have this question which has the information that needs to be queried in the weaviate vector database. 
        Transform the question into the optimized query and also the number of results returned in the form of JSON. The format should follow this rules:

        returned answer must be a JSON object with 2 propoerty.
        first property is "query" where the value holds an optimized query to be queried to the vector database and the data type is string.
        
        second property is "num_results" where the value is your reccomendation of how many document should be retrieved from vector database based on user's prompt. 
        if you think the user's prompt is complex and needs alot of information, the number of retrieved document from vector database should be bigger than a simple prompt.
        the value range of num_results is between 2 until 5.
        the data type of num_results property is number

        for example: 
        {{
            "query": "Sistem perpajakan di indonesia",
            "num_results": 2
        }}

        query property is the optimized query that will be queried to vector database to retrieve relevant documents.

        The data in my weaviate vector database contains data of indonesian's tax statutory rules written in indonesian language.

        Here is the question:
        <user_prompt>
    """
CHATBOT_PROMPT = """
    I have this question which you need to answer with indonesian language based on the given information.

    Don't try to answer the question if the answer is not exists in any of relevant informations below.
    here is the relevant informations:
    <relevant_informations>

    Here is the question:
    <user_prompt>
"""

def populate_vector_database():
    class_obj = {
        "class": weaviate_schema_name,
        "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
        }
    }

    if(weaviate_client.schema.exists(weaviate_schema_name) == False):
        print("schema: " + weaviate_schema_name + " not found. creating new class.")
        weaviate_client.schema.create_class(class_obj)
    else:
        print("schema: " + weaviate_schema_name + " exists.")

    # resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
    # data = json.loads(resp.text)  # Load data

    print(f"Starting to populate data to vector database.")

    # populate_from_file("./uu no 7 tahun 2021 - perpajakan.pdf")
    populate_from_file("./undang undang no 58 tahun 2023 dengan skema tarif.pdf")
    populate_from_file("./UU Nomor 36 Tahun 2008.pdf")
    # populate_from_file("./Sejarah Indonesia Masa Kemerdekaan.pdf")
    # populate_from_file("./Paket Kelas 9 IPA.pdf")

    print(f"all files populated successfully.")
    return "Processed successfully"

def populate_from_file(path):
    print(f"populating from file: " + path)
    
    text = parser.from_file(path)
    text_splitter = CharacterTextSplitter(chunk_size=8191, chunk_overlap=200)
    chunks = text_splitter.create_documents([text["content"]])
    print(f"chunks created. prepare to populate vector db")

    weaviate_client.batch.configure(batch_size=150)  # Configure batch
    with weaviate_client.batch as batch:  # Initialize a batch process
        for i, d in enumerate(chunks):  # Batch import data
            print(f"importing batch: {i+1}")
            # print(d.page_content)
            properties = {
                "content": d.page_content
            }
            batch.add_data_object(
                data_object=properties,
                class_name=weaviate_schema_name
            )
    
    print("file: " + path + " processed.")

def semantic_search(prompt, query_limit_reccomendation):
    response = (
        weaviate_client.query
        .get(weaviate_schema_name, ["content"])
        .with_near_text({"concepts": [prompt]})
        .with_limit(query_limit_reccomendation)
        .do()
    )

    response_data = response["data"]['Get'][weaviate_schema_name]
    response_data = map(lambda x: x['content'].replace('\n', ''), response_data)
    response_data = list(response_data);

    return response_data

def gpt_chat_completion(prompt, role, is_weaviate_query):    
    messages = [ 
        {
            "role": "system", 
            "content": role
        },
        {
            "role": "user", 
            "content": prompt
        }, 
    ]

    if(is_weaviate_query):
        return gpt_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            response_format={ "type": "json_object" }
        ).choices[0].message.content

    return gpt_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages,
    ).choices[0].message.content

def ask_gpt(user_prompt):
    weaviate_prompt = WEAVIATE_PROMPT.replace("<user_prompt>", user_prompt)
    response = gpt_chat_completion(weaviate_prompt, WEAVIATE_ROLE, True) 

    response_obj = json.loads(response)

    print("=== Start processing semantic search: " + response_obj['query'] + ". with limit: " + str(response_obj['num_results']))
    relevant_data = semantic_search(response_obj['query'], response_obj['num_results'])
    relevant_data = ".".join(relevant_data)

    print("=== Retrieved relevant data: " + relevant_data)

    chatbot_prompt = CHATBOT_PROMPT.replace("<relevant_informations>", relevant_data).replace("<user_prompt>", user_prompt)

    
    print("=== Start asking GPT for: " + user_prompt)
    response = gpt_chat_completion(chatbot_prompt, CHATBOT_ROLE, False)

    print("=== response from gpt: " + response)

    return response

# 1 - 10
# ask_gpt("apa itu PTKP ?")
# ask_gpt("apa itu Tarif Efektif Rata Rata bulanan berdasarkan uu nomor 53 tahun 2023?")
# ask_gpt("apa perbedaan sistem perpajakan antara memakai tarif efektif rata rata dengan aturan sebelum memakai tarif efektif rata rata dalam uu no 58 tahun 2023 ?")
# ask_gpt("berdasarkan undang undang no 58 tahun 2023, bagaimana cara menghitung pajak penghasilan dengan tarif efektif rata rata ?")
# ask_gpt("Berikan saya persentase pajak untuk setiap kategori yang ada dalam tarif efektif rata rata berdasarkan uu no 58 tahun 2023")
# ask_gpt("berdasarkan undang undang no 58 tahun 2023, berapa pajak yang dikenakan jika saya termasuk dalam kategori A dan memiliki penghasilan bruto 5.650.000 rupiah ?")
# ask_gpt("Apa itu golongan wajib pajak orang pribadi ?")
# ask_gpt("berdasarkan undang undang no 58 tahun 2023, jika saya belum memiliki status belum kawin dan tidak memiliki tanggungan, saya termasuk kedalam golongan yang mana ?")
# ask_gpt("berdasarkan undang undang no 58 tahun 2023, jika saya tidak kawin dan memiliki tanggungan sebanyak 2 orang. termasuk kedalam golongan manakah saya ?")
# ask_gpt("berikan saya kategori untuk tarif efektif bulanan berdasarkan ayat 2 undang undang no 58 tahun 2023")

# 11 - 20
# ask_gpt("apa saja bentuk natura atau kenikmatan berdasarkan uu no 7 tahun 2021 ?")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")
# ask_gpt("")

# populate_vector_database()