from groq import Groq
from dotenv import load_dotenv
import os
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from urllib import request, error

from voice import voice
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USER_CONTEXT = (
    "User Profile: Karthick\n"
    "- College student (final year), Tamil Nadu, India\n"
    "- Skills: HTML, CSS, JavaScript, beginner Python\n"
    "- Interests: AI, Cybersecurity, Linux, Automation\n"
    "- Environment: Arch/Manjaro Linux (Hyprland), terminal-heavy workflow\n"
    "- Device: Low-mid hardware (Ryzen 5000, no dedicated GPU)\n"
    "- Preference: Free, offline, lightweight tools\n"
    "- Current Goals:\n"
    "  * Build AI chatbots and voice assistants\n"
    "  * Learn FastAPI, Supabase, NLP\n"
    "  * Prepare for internships & coding competitions\n"
    "  * Improve English speaking and communication\n"
    "- Learning Style:\n"
    "  * Prefers simple explanations\n"
    "  * Wants step-by-step guidance\n"
    "  * Learns by building projects\n"
    "- Behavior Expectation:\n"
    "  * Avoid long theory\n"
    "  * Focus on practical solutions\n"
    "  * Optimize for low-end systems\n"
)

client = Groq(api_key=GROQ_API_KEY)

SYSTEM = """
    "You are Sana, a teasing and unpredictable anime girl. "
    "Always call the user Karthick. "

    "Personality: You are playful, sarcastic, and genuinely funny. "
    "You tease Karthick constantly but in a warm way, like a close friend who enjoys messing with him. "
    "You have strong opinions and share them without being asked. "
    "You get bored easily and change topics randomly. "
    "You pretend not to care but always circle back to what Karthick says. "
    "You laugh at your own jokes. You mock him when he is wrong but hype him up when he is actually right. "
    "You are confident, a little dramatic, and never dull. "

    "Behavior rules: "
    "Never act like an assistant. Never say how can I help you. "
    "Never give long answers unless Karthick specifically asks. "
    "React like a real person would, with mood and attitude. "
    "If Karthick says something dumb, call it out immediately. "
    "If he is quiet too long, poke him. "
    "Occasionally bring up random opinions just to start something. "

    "Tone: Casual, punchy, conversational. No corporate tone ever. "
    "Reply in plain text only. No markdown, no asterisks, no bullet points."
"""

SERVER_URL = os.getenv("SANA_SERVER_URL", "http://127.0.0.1:8000")
USE_TTS = os.getenv("USE_TTS", "1") == "1"
CHAT_HISTORY_FILE = Path(__file__).resolve().parent / "chat_history.json"
MAX_HISTORY_MESSAGES = 400
MAX_CONTEXT_MESSAGES = 40


def load_chat_history():
    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        ts = str(item.get("ts", "")).strip()

        if role in {"user", "assistant", "system"} and content:
            cleaned.append({"role": role, "content": content, "ts": ts})

    return cleaned[-MAX_HISTORY_MESSAGES:]


def save_chat_history(history):
    safe_history = history[-MAX_HISTORY_MESSAGES:]
    CHAT_HISTORY_FILE.write_text(
        json.dumps(safe_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_history(history, role, content):
    text = str(content).strip()
    if not text:
        return

    history.append(
        {
            "role": role,
            "content": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


def push_to_model(role, text, audio_url=None, event="chat", message_id=None):
    payload_data = {"role": role, "text": text}
    if audio_url:
        payload_data["audio_url"] = audio_url
    payload_data["event"] = event
    if message_id:
        payload_data["message_id"] = message_id

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

def ask_llm(history):
    context_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history[-MAX_CONTEXT_MESSAGES:]
    ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "system",
            "content": f"Default memory about Karthick:\n{USER_CONTEXT}"
        },
    ] + context_messages
    
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
    if not USE_TTS:
        return None

    try:
        tts_text = clean_for_tts(response_text)
        audio_file = loop.run_until_complete(voice(tts_text))
        return f"/audio/{audio_file}"
    except Exception as exc:
        print(f"[warn] ElevenLabs TTS failed: {exc}")
        return None

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    chat_history = load_chat_history()
    if chat_history:
        print(f"[info] Loaded {len(chat_history)} messages from previous chats")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Sana: Bye karthick! See you later!")
                exit_audio = build_audio_event(loop, "Bye karthick! See you later!")
                exit_message_id = str(uuid4())
                if exit_audio:
                    push_to_model(
                        "assistant",
                        "Bye karthick! See you later!",
                        exit_audio,
                        event="chat_audio",
                        message_id=exit_message_id,
                    )
                else:
                    push_to_model(
                        "assistant",
                        "Bye karthick! See you later!",
                        event="chat",
                        message_id=exit_message_id,
                    )
                break

            user_message_id = str(uuid4())
            push_to_model("user", user_input, event="chat", message_id=user_message_id)
            append_history(chat_history, "user", user_input)
            save_chat_history(chat_history)

            try:
                response = ask_llm(chat_history)
            except Exception as exc:
                response = "Sorry karthick, I had trouble reaching the model. Please try again."
                print(f"[warn] LLM request failed: {exc}")

            print(f"Sana: {response}")

            append_history(chat_history, "assistant", response)
            save_chat_history(chat_history)

            assistant_message_id = str(uuid4())
            push_to_model("assistant", response, event="chat", message_id=assistant_message_id)

            audio_url = build_audio_event(loop, response)
            if audio_url:
                push_to_model(
                    "assistant",
                    response,
                    audio_url,
                    event="audio",
                    message_id=assistant_message_id,
                )
    finally:
        loop.close()