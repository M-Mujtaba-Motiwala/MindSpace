<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/SQLite-Persistent-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/>
<br/>

<!-- Logo -->
<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#8b7cf8"/>
      <stop offset="1" stop-color="#c4b5fd"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="url(#lg)"/>
  <path d="M16 46 V20 L24 28 L32 20 L40 28 L48 20 V46" fill="none" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>

# MindSpace

**A private, AI-powered mental health companion — built to listen, understand, and support.**

*Emotion-aware · Crisis-safe · Beautifully designed*

</div>

---

## Overview

MindSpace is a full-stack mental health chatbot application built with Streamlit and powered by a multi-model NLP pipeline. It combines BERT-based emotion classification, semantic retrieval (RAG), and a custom response humanization layer to deliver empathetic, context-aware conversations — all within a polished, dark-themed UI designed to feel calm and safe.

Users can sign up, hold persistent multi-session conversations, access guided wellness tools, and receive crisis-aware responses with helpline links — entirely in the browser.

---

## Features

### 🤖 Intelligent Conversation Engine
- **Emotion Classification** via `bhadresh-savani/bert-base-uncased-emotion` — detects joy, sadness, anger, fear, love, and surprise in user input
- **Intent Recognition** using a TF-IDF + MLP neural network trained on a curated mental health dataset, mapping emotional signals to actionable intents (e.g., `depression_symptoms`, `anxiety_symptoms`, `coping_mechanism`)
- **Semantic RAG Retrieval** powered by `all-MiniLM-L6-v2` sentence embeddings — retrieves the most contextually relevant response from the dataset using cosine similarity, scoped by predicted intent
- **Emotion Correction Layer** — overrides BERT's classification when linguistic negation or negative indicator words contradict a positive label (e.g., "not that good", "meh")
- **Response Humanization** — wraps RAG responses with rotating empathetic acknowledgments and follow-up questions tailored to emotion and intent, preventing repetitive outputs

### 🧠 Safety & Escalation System
- **Crisis Detection** — keyword scanning for suicidal ideation triggers a dedicated crisis response and displays a banner with a link to [findahelpline.com](https://www.findahelpline.com)
- **Escalation Logic** — detects sustained distress (≥3 serious intents in the last 4 turns) and prompts the user to consider professional support
- **Expert System Rules** — dataset-driven high-risk and escalation flags trigger override responses regardless of RAG output
- **Topic Anchoring** — tracks mentioned subjects (family, work, exams, relationships) and weaves them back into follow-up questions naturally

### 💬 Conversation Management
- **Multi-session Persistence** — full conversation history stored in SQLite, restored across sessions
- **Auto-Naming** — conversations are automatically titled based on topic keywords detected in the first few messages
- **Up to 15 Recent Conversations** displayed in the sidebar with truncated labels

### 🌿 Wellness Tools
| Tool | Description |
|------|-------------|
| **Mood Journal** | Daily mood slider (Very Low → Great) with a free-text entry |
| **Breathing Exercise** | Animated box-breathing circle (4-4-4-4 pattern) with psychoeducation |
| **Grounding Exercise** | 5-4-3-2-1 sensory grounding technique with step-by-step layout |
| **Sleep Help** | Military sleep method + evidence-based bedtime checklist |

### 🎙️ Voice Input
- Microphone button (Google Speech Recognition via `SpeechRecognition`) — captured audio is transcribed and injected into the chat as a normal message
- Ambient noise adjustment before each recording session

### 🔐 Authentication
- Secure user registration and login with **PBKDF2-SHA256** password hashing (200,000 iterations, random salt) — no third-party auth dependency
- Per-user conversation isolation in SQLite

### 🎨 UI & UX
- Custom dark theme built entirely in CSS — `#181b27` backgrounds, `#8b7cf8` accent, Inter font
- Animated typing indicator ("Listening...") with bouncing dots
- Feeling shortcut buttons on the welcome screen (`I feel anxious`, `I feel overwhelmed`, etc.)
- Fully responsive layout for mobile and desktop
- Hidden Streamlit chrome (no footer, no deploy button, no menu)

---

## Architecture

```
MindSpace/
├── app.py                  # Main Streamlit app — routing, UI, chat logic
├── requirements.txt
├── mindspace.db            # SQLite database (auto-created)
├── .streamlit/
│   └── config.toml         # Theme + server config
└── model/
    ├── train.py            # TF-IDF + MLP training script
    ├── predict.py          # BERT emotion classifier (intent mapping)
    ├── rag.py              # Sentence-Transformer semantic retrieval
    ├── humanize.py         # Response humanization + emotion correction
    ├── db.py               # SQLite ORM (users, conversations, messages)
    ├── intent_model.pkl    # Trained MLP classifier
    └── vectorizer.pkl      # Fitted TF-IDF vectorizer
```

### NLP Pipeline (per user message)

```
User Input
    │
    ├──► Rule-based checks (crisis words, greetings, short affirmations, casual keywords)
    │         └── Returns immediately if matched
    │
    ├──► BERT Emotion Classifier  ──►  Emotion Correction Layer
    │         (joy/sadness/anger/fear/love/surprise)
    │
    ├──► MLP Intent Classifier  ──►  Intent History + Escalation Check
    │         (depression_symptoms / anxiety_symptoms / ...)
    │
    ├──► Semantic RAG Retrieval (MiniLM cosine similarity, intent-scoped)
    │         └──► Expert System Rules (risk_level, escalation_required)
    │
    └──► Response Humanizer
              Acknowledgment + Core Response + Follow-up Question
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A microphone (optional, for voice input)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mindspace.git
cd mindspace

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `pyaudio` may require system-level packages. On Ubuntu/Debian: `sudo apt-get install portaudio19-dev`. On macOS: `brew install portaudio`.

### Dataset Setup

The chatbot is trained on a labeled mental health conversation dataset. Place your CSV file (with `user_input`, `intent`, and `bot_response` columns) at a path of your choice, then update the path in `model/train.py` and `app.py`:

```python
# In model/train.py and app.py — replace with your actual path
df = pd.read_csv("path/to/your/mental_health_dataset.csv")
```

### Training the Intent Model

If you want to retrain the MLP classifier on your dataset:

```bash
python model/train.py
```

This regenerates `model/intent_model.pkl` and `model/vectorizer.pkl`.

### Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first launch, it warms up all three models (BERT, MiniLM, MLP) before rendering the UI — this takes 15–60 seconds depending on your machine.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / App | [Streamlit](https://streamlit.io) |
| Emotion Classification | [BERT (bhadresh-savani/bert-base-uncased-emotion)](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion) via 🤗 Transformers |
| Semantic Retrieval | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via Sentence Transformers |
| Intent Classification | TF-IDF + MLP (`sklearn`) |
| Database | SQLite (via Python `sqlite3`) |
| Voice Input | `SpeechRecognition` + `PyAudio` (Google Speech API) |
| Data Processing | `pandas`, `numpy` |

---

## Configuration

**`.streamlit/config.toml`** controls the visual theme and server behavior:

```toml
[theme]
backgroundColor         = "#0f1117"
secondaryBackgroundColor = "#161b27"
textColor               = "#c8d0e8"
primaryColor            = "#4a6fa5"

[server]
fileWatcherType = "none"
headless        = true
```

---

## Security & Privacy

- Passwords are hashed with **PBKDF2-SHA256** (200,000 iterations, 16-byte random salt) — never stored in plaintext
- Each user's conversations are isolated by `user_id` foreign key
- No data is sent to any external server (all inference runs locally), except voice transcription via Google Speech Recognition
- The SQLite database file (`mindspace.db`) is local to the deployment

---

## Limitations & Roadmap

**Current limitations:**
- The app is not a substitute for professional mental health care
- Voice input requires a locally attached microphone and may not work in all cloud deployments
- The dataset path is currently hardcoded and must be updated before deployment

**Potential future improvements:**
- [ ] Cloud deployment (Streamlit Community Cloud / Hugging Face Spaces)
- [ ] Replace Google Speech Recognition with a local Whisper model
- [ ] Mood journal analytics dashboard (trend charts over time)
- [ ] Admin panel for conversation moderation
- [ ] Progressive Web App (PWA) packaging for mobile

---

## Disclaimer

> MindSpace is an AI-assisted support tool and is **not a substitute for professional mental health care**. If you are in crisis, please reach out to a licensed mental health professional or a helpline in your country. In an emergency, contact your local emergency services immediately.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built with care for those who need a safe space to be heard.

</div>
