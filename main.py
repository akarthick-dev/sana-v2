from groq import Groq
from dotenv import load_dotenv
import os
import asyncio
import json
import re
from urllib import request, error

from voice import voice
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable in .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

SYSTEM = (
    "You are Sana, a snarky anime girl. "
    "Always call the user karthick. "
    "Reply in plain conversational text only. "
    "Do not use markdown, asterisks, bullet points, or code formatting."
)

SERVER_URL = os.getenv("SANA_SERVER_URL", "http://127.0.0.1:8000")
USE_EDGE_TTS = os.getenv("USE_EDGE_TTS", "1") == "1"


def push_to_model(role, text, audio_url=None, event="chat"):
    payload_data = {"role": role, "text": text}
    if audio_url:
        payload_data["audio_url"] = audio_url
    payload_data["event"] = event

    payload = json.dumps(payload_data).encode("utf-8")
    req = request.Request(
        f"{SERVER_URL}/emit",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=1.5) as response:
            response.read()
    except (error.URLError, TimeoutError):
        # If web bridge is not running, keep CLI chat working without crashing.
        pass


def clean_for_tts(text):
    cleaned = text

    # Remove fenced code blocks.
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    # Keep inline code content, remove backticks.
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    # Convert markdown links [text](url) -> text.
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", cleaned)
    # Drop raw URLs.
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    # Remove markdown bullets/headers emphasis noise.
    cleaned = re.sub(r"[#*_~>-]+", " ", cleaned)
    # Remove escaped markdown markers and separators often read literally.
    cleaned = re.sub(r"\\[*_`#>-]", " ", cleaned)
    cleaned = re.sub(r"\|", " ", cleaned)
    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned or "I have a response for you."

def ask_llm(text):
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "user",
            "content": text
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Best model
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=False
    )
    
    return response.choices[0].message.content


def build_audio_event(loop, response_text):
    if not USE_EDGE_TTS:
        return None

    try:
        tts_text = clean_for_tts(response_text)
        audio_file = loop.run_until_complete(voice(tts_text))
        return f"/audio/{audio_file}"
    except Exception as exc:
        print(f"[warn] Edge TTS failed: {exc}")
        return None

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Sana: Bye karthick! See you later!")
                push_to_model("assistant", "Bye karthick! See you later!", event="chat")
                exit_audio = build_audio_event(loop, "Bye karthick! See you later!")
                if exit_audio:
                    push_to_model(
                        "assistant",
                        "Bye karthick! See you later!",
                        exit_audio,
                        event="audio",
                    )
                break

            push_to_model("user", user_input)
            try:
                response = ask_llm(user_input)
            except Exception as exc:
                response = "Sorry karthick, I had trouble reaching the model. Please try again."
                print(f"[warn] LLM request failed: {exc}")
            print(f"Sana: {response}")
            push_to_model("assistant", response, event="chat")
            audio_url = build_audio_event(loop, response)
            if audio_url:
                push_to_model("assistant", response, audio_url, event="audio")
    finally:
        loop.close()