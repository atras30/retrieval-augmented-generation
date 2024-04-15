from flask import Flask, request
from flask_cors import CORS, cross_origin
from gpt import ask_gpt, populate_vector_database

app = Flask(__name__)
cors = CORS(app, origins=['http://localhost:3000','http://localhost:3000'])

@app.route("/")
def index():
    user_prompt = request.args['search']
    gpt_response = ask_gpt(user_prompt)
    return gpt_response

@app.route("/populate")
def populate():
    print(f"Hitting populate endpoint")
    populate_vector_database()
    return "Success"

if __name__ == "__main__":
    app.run(debug=True)
