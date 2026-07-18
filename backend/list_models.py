import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print("EMBEDDING MODEL:", m.name)
    if "generateContent" in m.supported_generation_methods:
        print("CHAT MODEL:", m.name)