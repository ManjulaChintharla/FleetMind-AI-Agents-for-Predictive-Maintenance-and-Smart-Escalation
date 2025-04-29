# 🚗 Contoso SmartFleet Assistant – Azure OpenAI + Azure AI Search

from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
#from azure.search.documents.indexes.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import re

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)

# Azure OpenAI setup (new SDK)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

embedding_deployment = os.getenv("AZURE_OPENAI_embedding_DEPLOYMENT")
chat_model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Azure AI Search setup
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("message", "")

    try:
        # Get embedding
        embedding_response = client.embeddings.create(
            model=embedding_deployment,
            input=[question]
        )
        embedding = embedding_response.data[0].embedding

        # Azure AI Search vector query
        vector_query = VectorizedQuery(vector=embedding, fields="embedding")
        results = search_client.search(
            search_text=" ", 
            vector_queries=[vector_query]
        )

        top_result = next(results, None)
        context = top_result["content"] if top_result else "No relevant documents found."

        # Prompt for GPT model
        prompt = f"""
You are a concise and intelligent fleet maintenance assistant. Use the context below to answer briefly:

Context:
{context}

Question: {question}
Respond in bullet points with 2–3 actionable tips. Include cost estimation next to each tip if available.
"""

        # Chat completion using GPT model
        response = client.chat.completions.create(
            model=chat_model_deployment,
            messages=[
                {"role": "system", "content": "You're a helpful fleet assistant. Reply clearly with tips and cost."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )

        answer = response.choices[0].message.content

        # Highlight cost figures like $100–$500
        highlighted = re.sub(r"(\$\d+[kK]?[\+\-]?\s?(–|-)?\s?\$?\d*[kK]?)", r"<span class='cost-highlight'>\1</span>", answer)
        return jsonify({"response": highlighted})

    except Exception as e:
        return jsonify({"response": f"⚠️ An error occurred:\n\n{str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
