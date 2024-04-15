from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
chat_gpt_api_key = os.getenv("CHAT_GPT_API_KEY")
client = OpenAI(api_key=chat_gpt_api_key)

messages = [ 
    {
        "role": "system", 
        "content": "You are a intelligent assistant."
    } 
]

while True: 
	message = input("User : ") 
	if message: 
		messages.append( 
			{"role": "user", "content": message}, 
		) 
		chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages) 

	reply = chat.choices[0].message.content 
	print(f"ChatGPT: {reply}") 
	messages.append({"role": "assistant", "content": reply})
