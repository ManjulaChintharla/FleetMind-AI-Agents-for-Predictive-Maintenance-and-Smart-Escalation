from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import re
from agent_escalation import evaluate_escalation, trigger_logic_app

# Load .env
load_dotenv()

app = Flask(__name__)

# Azure OpenAI setup
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
        # Step 1: Generate embedding
        embedding_response = client.embeddings.create(
            model=embedding_deployment,
            input=[question]
        )
        embedding = embedding_response.data[0].embedding

        # Step 2: Vector search
        vector_query = VectorizedQuery(vector=embedding, fields="embedding")
        results = search_client.search(search_text=" ", vector_queries=[vector_query])
        top_result = next(results, None)
        context = top_result["content"] if top_result else "No relevant documents found."

        # Step 3: Prompt with improved instruction
        prompt = f"""
You are an AI-powered fleet assistant. Your job is to provide smart diagnostic tips and cost estimates for vehicle issues.

Only include '**ESCALATE**' at the end of your response if:
- The issue involves fire, fuel leaks, explosions, major overheating, engine failure, or brake failure
- OR the situation clearly requires immediate human intervention

Otherwise, just provide helpful, concise suggestions and estimated repair costs.

Context:
{context}

User issue:
{question}
"""

        completion = client.chat.completions.create(
            model=chat_model_deployment,
            messages=[
                {"role": "system", "content": "You are a smart fleet assistant that helps diagnose and recommend repairs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )

        answer = completion.choices[0].message.content

        # Step 4: Evaluate escalation if **ESCALATE** is present
        if "**ESCALATE**" in answer.upper():
            decision = evaluate_escalation(question, context)
            if decision == "ESCALATE":
                triggered = trigger_logic_app(question, context)
                cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
                cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
                highlighted = re.sub(
                    r"(\$\d+[kK]?[\+\-]?\s?(–|-)?\s?\$?\d*[kK]?)",
                    r"<span class='cost-highlight'>\1</span>",
                    cleaned
                )
                if triggered:
                    return jsonify({
                        "response": highlighted + "<br><br> <b>Escalated to human engineer.</b> You’ll receive an update via email shortly."
                    })
                else:
                    return jsonify({
                        "response": highlighted + "<br><br> Escalation failed while calling Logic App."
                    })

        # Step 5: Standard response (clean + highlight)
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        highlighted = re.sub(
            r"(\$\d+[kK]?[\+\-]?\s?(–|-)?\s?\$?\d*[kK]?)",
            r"<span class='cost-highlight'>\1</span>",
            cleaned
        )

        return jsonify({"response": highlighted})

    except Exception as e:
        return jsonify({"response": f" Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
