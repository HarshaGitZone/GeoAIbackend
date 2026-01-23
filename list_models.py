import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

models = list(client.models.list())
print("MODELS AVAILABLE TO THIS KEY:\n")

for m in models:
    print(m.name)
