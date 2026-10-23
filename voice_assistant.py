"""
Voice Assistant
---------------
Pipeline:  Microphone -> ElevenLabs Speech-to-Text -> LLM (Claude) -> ElevenLabs Text-to-Speech -> Speakers

Say "exit", "quit" or "goodbye" to stop.
"""

import io
import os
import sys
import time

import anthropic
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

# ----------------------------- Configuration -----------------------------
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George" default voice
TTS_MODEL = "eleven_multilingual_v2"
STT_MODEL = "scribe_v1"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

TTS_SAMPLE_RATE = 22050          # matches output_format "pcm_22050"
MIC_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01         # raise if background noise stops recording from ending
SILENCE_SECONDS = 1.2            # how long of silence ends your turn
MAX_RECORD_SECONDS = 30
BLOCK_SECONDS = 0.1

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so keep them "
    "short (1-3 sentences), conversational, and free of markdown, lists or emojis."
)

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "stop"}


# ------------------------------ Audio input ------------------------------
def record_until_silence() -> np.ndarray | None:
    """Record from the mic. Starts when you speak, stops after a pause."""
    block = int(MIC_SAMPLE_RATE * BLOCK_SECONDS)
    chunks: list[np.ndarray] = []
    started = False
    silent_blocks = 0
    max_blocks = int(MAX_RECORD_SECONDS / BLOCK_SECONDS)
    silence_limit = int(SILENCE_SECONDS / BLOCK_SECONDS)

    print("🎤 Listening...")
    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for _ in range(max_blocks):
            data, _ = stream.read(block)
            volume = float(np.sqrt(np.mean(data**2)))

            if volume > SILENCE_THRESHOLD:
                started = True
                silent_blocks = 0
            elif started:
                silent_blocks += 1

            if started:
                chunks.append(data)
                if silent_blocks >= silence_limit:
                    break

    if not chunks:
        return None
    return np.concatenate(chunks)


# --------------------------- Speech-to-Text ------------------------------
def transcribe(eleven: ElevenLabs, audio: np.ndarray) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, MIC_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    buffer.name = "speech.wav"

    result = eleven.speech_to_text.convert(file=buffer, model_id=STT_MODEL)
    return result.text.strip()


# ------------------------------ LLM reply --------------------------------
def generate_reply(llm: anthropic.Anthropic, history: list[dict]) -> str:
    response = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


# --------------------------- Text-to-Speech ------------------------------
def speak(eleven: ElevenLabs, text: str) -> None:
    stream = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=TTS_MODEL,
        output_format="pcm_22050",
    )
    pcm_bytes = b"".join(stream)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(samples, TTS_SAMPLE_RATE)
    sd.wait()


# -------------------------------- Main -----------------------------------
def main() -> None:
    if not ELEVEN_API_KEY or not ANTHROPIC_API_KEY:
        sys.exit("Missing API keys. Copy .env.example to .env and fill in both keys.")

    eleven = ElevenLabs(api_key=ELEVEN_API_KEY)
    llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    history: list[dict] = []

    print("Voice assistant ready. Say 'goodbye' to quit.\n")
    speak(eleven, "Hi! How can I help you today?")

    while True:
        try:
            audio = record_until_silence()
            if audio is None:
                continue

            user_text = transcribe(eleven, audio)
            if not user_text:
                continue
            print(f"🧑 You: {user_text}")

            if user_text.lower().strip(" .!?") in EXIT_WORDS:
                speak(eleven, "Goodbye!")
                break

            history.append({"role": "user", "content": user_text})
            reply = generate_reply(llm, history)
            history.append({"role": "assistant", "content": reply})
            print(f"🤖 Assistant: {reply}\n")

            speak(eleven, reply)

            # keep memory bounded so requests stay fast and cheap
            history = history[-20:]

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:  # network / API errors shouldn't crash the loop
            print(f"⚠️  Error: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main()
