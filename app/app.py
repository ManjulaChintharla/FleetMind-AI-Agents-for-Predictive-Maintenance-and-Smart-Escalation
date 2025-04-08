# app.py

import os
import openai
import requests
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load env variables
load_dotenv(dotenv_path=".env")

# Setup: AML
aml_endpoint = os.getenv("AZURE_ML_ENDPOINT")
aml_api_key = os.getenv("AZURE_ML_API_KEY")

# Setup: Azure AI Search
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_key = os.getenv("AZURE_SEARCH_KEY")

# Setup: Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
embedding_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
chat_model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Init Search Client
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

# ---- Function: Get AML Prediction ----
def get_aml_prediction(input_data):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aml_api_key}"
    }
    response = requests.post(aml_endpoint, headers=headers, json=input_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f" AML Error {response.status_code}: {response.text}")
        return None

# ---- Function: Get Embedding ----
def get_embedding(text: str):
    response = openai.Embedding.create(
        input=[text],
        deployment_id=embedding_deployment
    )
    return response['data'][0]['embedding']

# ---- Function: Vector Search + GPT Answer ----
def search_index(query_text, top_k=1):
    try:
        vector = get_embedding(query_text)
        vector_query = VectorizedQuery(
            vector=vector,
            fields="embedding"
        )
        results = search_client.search(
            search_text=" ",  # Required
            vector_queries=[vector_query]
        )
        best_result = next(results, None)

        if best_result:
            document_content = best_result.get("content", "")
            prompt = f"""
You are a helpful assistant for fleet vehicle maintenance.

Use the following context from a document to answer the question.

Context:
\"\"\"
{document_content}
\"\"\"

Question: {query_text}
Answer:"""

            completion = openai.ChatCompletion.create(
                deployment_id=chat_model_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for fleet maintenance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            answer = completion['choices'][0]['message']['content']
            print(f"\n GPT Answer for: \"{query_text}\"\n")
            print(answer)
        else:
            print(" No relevant results found in search.")

    except Exception as e:
        print("Error during vector search + GPT:", e)

# ---- Function: Full Flow ----
def run_prediction_and_search(input_vehicle_data):
    prediction = get_aml_prediction(input_vehicle_data)

    if prediction:
        print(" Prediction result:", prediction)
        maintenance_flag = prediction.get("maintenanceflag")

        if maintenance_flag == 1:
            query = "What are the best practices to prevent major vehicle breakdowns?"
        else:
            query = "How do I continue maintaining a healthy vehicle engine?"

        search_index(query)
    else:
        print(" Prediction failed. Skipping search.")

# ---- Example Input + Run ----
if __name__ == "__main__":
    example_input = {
        "data": [
            {
                "enginehealth": 0.8,
                "vehiclespeedsensor": 60,
                "enginecoolanttemp": 95,
                "enginerpm": 3500,
                "massairflowrate": 22.0,
                "speedgps": 59,
                "litresper100kminst": 8.5,
                "co2ingperkminst": 140,
                "triptimejourney": 25
            }
        ]
    }

    run_prediction_and_search(example_input)
