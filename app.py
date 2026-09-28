import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder="templates")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PROMPT_PATH = BASE_DIR / "chatbot_config.txt"


def system_prompt():
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "You are AgriSmart AI. Answer only agriculture and AgriSmart AI questions; politely decline unrelated requests."


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Please enter a question."), 400
    if len(message) > 6000:
        return jsonify(error="Please keep your question under 6,000 characters."), 400
    if not API_KEY:
        return jsonify(error="GEMINI_API_KEY is not configured. Add your API key to the .env file."), 503

    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt(),
                temperature=0.4,
                max_output_tokens=1200,
            ),
        )
        return jsonify(reply=(response.text or "Please try asking your question another way.").strip())
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify(error="The AI service is unavailable right now. Please try again shortly."), 502


if __name__ == "__main__":
    app.run()
