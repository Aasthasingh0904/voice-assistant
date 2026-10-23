# Voice Assistant (Python)

A conversational voice assistant: speak -> transcribe -> generate reply -> speak back.

**Pipeline:** Microphone -> ElevenLabs Speech-to-Text -> Claude (LLM) -> ElevenLabs Text-to-Speech -> Speakers

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API keys
python voice_assistant.py
```
On Linux you may also need: `sudo apt install libportaudio2 libsndfile1`

## Usage
Speak after "Listening...". Pause for ~1 second and the assistant replies. Say "goodbye" to exit.

## Tuning
- `SILENCE_THRESHOLD`: raise it in noisy rooms.
- `SILENCE_SECONDS`: how long a pause ends your turn.
- `ELEVENLABS_VOICE_ID`: pick any voice from your ElevenLabs library.
