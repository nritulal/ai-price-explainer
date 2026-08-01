import sys
sys.path.insert(0, r"C:\Users\Ritu N Lal\PycharmProjects\ai-price-explainer")

from src.utils.constants import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")
response = model.generate_content("Hello, write a one-sentence greeting for a retail pricing demo.")
print(response.text)