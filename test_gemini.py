import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model_id = "gemini-3.5-flash" 
    
    messages = [
        {"role": "user", "content": "Tell me a long story about a dragon and a knight. Write at least 500 words."}
    ]
    
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

    print(f"Calling Gemini with model {model_id}...")
    response = client.models.generate_content(
        model=model_id,
        contents=gemini_contents,
        config={
            "temperature": 0.7,
            "max_output_tokens": 4000,
        },
    )
    
    print(f"DEBUG: response: {response}")
    print(f"Response text length: {len(response.text)}")
    print(f"First 100 chars: {response.text[:100]}")
    print(f"Last 100 chars: {response.text[-100:]}")
    
    if len(response.text) < 200:
        print("Response seems truncated!")
        print(f"Full response: {response.text}")

if __name__ == "__main__":
    test_gemini()
