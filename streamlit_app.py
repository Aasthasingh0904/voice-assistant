import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types

load_dotenv()


def get_secret(name, default=None):
    """Read from Streamlit secrets (deployed) or .env / environment (local)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ELEVEN_API_KEY = get_secret("ELEVENLABS_API_KEY")
VOICE_ID = get_secret("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
LLM_MODEL = get_secret("LLM_MODEL", "gemini-3.5-flash")

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Keep answers short (1-3 sentences) "
    "and conversational, since they will be read aloud."
)

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️")
st.title("🎙️ Voice Assistant")

if not GEMINI_API_KEY or not ELEVEN_API_KEY:
    st.error(
        "Missing API keys. Add GEMINI_API_KEY and ELEVENLABS_API_KEY to .env "
        "(local) or to the app's Secrets (Streamlit Cloud)."
    )
    st.stop()


@st.cache_resource
def get_clients():
    return genai.Client(api_key=GEMINI_API_KEY), ElevenLabs(api_key=ELEVEN_API_KEY)


gemini, eleven = get_clients()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user_text, bot_text)
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "last_reply_audio" not in st.session_state:
    st.session_state.last_reply_audio = None


def transcribe(audio_bytes: bytes) -> str:
    resp = gemini.models.generate_content(
        model=LLM_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Transcribe this audio exactly. Return only the transcript.",
        ],
    )
    return (resp.text or "").strip()


def reply_to(text: str) -> str:
    contents = []
    for user_msg, bot_msg in st.session_state.history:
        contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))
        contents.append(types.Content(role="model", parts=[types.Part(text=bot_msg)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
    resp = gemini.models.generate_content(
        model=LLM_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return (resp.text or "").strip()


def speak(text: str) -> bytes:
    stream = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(stream)


audio = st.audio_input("Press the mic, speak, then press stop")

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    # Streamlit reruns the script on every interaction; only process new recordings.
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        try:
            with st.spinner("Listening..."):
                user_text = transcribe(audio_bytes)
            if not user_text:
                st.warning("Couldn't hear anything. Please try again.")
            else:
                with st.spinner("Thinking..."):
                    answer = reply_to(user_text)
                with st.spinner("Speaking..."):
                    st.session_state.last_reply_audio = speak(answer)
                st.session_state.history.append((user_text, answer))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

if st.session_state.last_reply_audio:
    st.audio(st.session_state.last_reply_audio, format="audio/mp3", autoplay=True)

if st.session_state.history and st.button("Clear conversation"):
    st.session_state.history = []
    st.session_state.last_reply_audio = None
    st.rerun()