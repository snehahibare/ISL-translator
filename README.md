# 🤟 ISL Translator — Real-Time Indian Sign Language AI

<div align="center">

![ISL Translator Banner](https://img.shields.io/badge/ISL%20Translator-v3.0-6c5ce7?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xOCA2LjVjMCAuOC0uNiAxLjUtMS41IDEuNXMtMS41LS43LTEuNS0xLjVTMTUuNyA1IDE2LjUgNSAxOCA1LjcgMTggNi41eiIvPjwvc3ZnPg==)

**A production-grade, full-stack AI application that translates Indian Sign Language (ISL) gestures into spoken English sentences — in real-time.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-NLP-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

</div>

---

## 🌟 What Is This?

A **portfolio-worthy, end-to-end AI system** that bridges the communication gap between the Deaf/Hard-of-Hearing community and the general public. Point your webcam, perform an ISL sign, and watch the AI translate it into a natural English sentence — spoken aloud.

> **Deployed Stack:** MediaPipe → Deep Learning LSTM → FastAPI → Gemini AI → TTS → Premium Web UI

---

## ✨ Key Features

| Feature | Details |
|:--------|:--------|
| 🦴 **Live Hand Skeleton Overlay** | Real-time MediaPipe landmark visualization drawn directly on the camera feed |
| 🧠 **Deep Learning LSTM Model** | Custom bidirectional LSTM trained on 318+ ISL gestures — sequences of 30 frames |
| 📈 **Live Confidence Sparkline** | Real-time chart showing model confidence over the last 30 predictions |
| 🎯 **318+ Sign Recognition** | Comprehensive ISL vocabulary covering greetings, actions, objects, and more |
| ✨ **Gemini AI (NLP)** | Google Gemini 2.0 converts raw sign words into fluent, grammatical sentences |
| 🔊 **Text-to-Speech** | Speaks the translated sentence aloud using Web Speech API & pyttsx3 |
| 🎤 **Voice Commands** | Hands-free control — say "start", "stop", "translate", "speak" |
| 📊 **Session Analytics** | Interactive dashboard: signs/min, top sign, avg confidence, frequency chart |
| 📚 **Sign Reference Gallery** | Searchable modal listing all 318+ supported ISL signs |
| ⚡ **SVG Progress Ring** | Visual buffer fill indicator showing frame sequence loading |
| ⌨️ **Keyboard Shortcuts** | Full keyboard control: S / X / T / Space / M / G / A |
| 🗒️ **PDF + TXT Export** | Export your translation session transcript |
| 🌙 **Dark / Light Mode** | Smooth animated theme toggle with glassmorphism UI |
| 🎓 **Onboarding Tour** | Built-in 5-step guided tour for first-time users |

---

## 🛠️ Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Computer Vision** | OpenCV, Google MediaPipe Holistic (258 keypoints) |
| **Deep Learning** | TensorFlow / Keras — Bidirectional LSTM |
| **Backend API** | FastAPI, Uvicorn, WebSockets |
| **Generative AI** | Google Gemini 2.0 Flash (NLP sentence builder) |
| **Text-to-Speech** | pyttsx3 (offline) + Web Speech API (browser) |
| **Frontend** | HTML5, CSS3 (Space Grotesk, glassmorphism), Vanilla JS |
| **Canvas APIs** | Hand skeleton (WebGL canvas), Sparkline chart, Progress ring (SVG) |

---

## 🔮 How It Works

```
┌─────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Webcam     │───▶│  MediaPipe        │───▶│  LSTM Model       │
│  (Browser)  │    │  258 keypoints    │    │  30-frame buffer  │
└─────────────┘    └───────────────────┘    └────────┬──────────┘
                         │                           │
                         │ landmarks →               │ sign →
                         ▼                           ▼
                   ┌─────────────┐         ┌───────────────────┐
                   │  Skeleton   │         │  Gemini AI (NLP)  │
                   │  Overlay    │         │  Sentence Builder │
                   └─────────────┘         └────────┬──────────┘
                                                    │
                                                    ▼
                                           ┌───────────────────┐
                                           │  TTS Engine       │
                                           │  Speaks Aloud 🔊  │
                                           └───────────────────┘
```

1. **Browser** captures webcam frames → sends via WebSocket
2. **MediaPipe Holistic** extracts 258 body + hand keypoints per frame
3. **30-frame sequence buffer** is filled (visible via progress ring)
4. **LSTM Model** classifies the gesture sequence → predicts ISL sign
5. **Signs accumulate** as word chips in the UI
6. **Gemini AI** converts word sequence → natural English sentence
7. **TTS Engine** speaks the sentence aloud



---

## 📁 Project Structure

```
ISL-Translator/
├── backend/
│   ├── api_server.py           # FastAPI + WebSocket server (main app entry)
│   ├── nlp_engine.py           # Google Gemini AI sentence builder
│   ├── tts_engine.py           # Text-to-Speech engine (pyttsx3)
│   ├── data_collection.py      # Custom ISL dataset recorder (MediaPipe)
│   ├── extract_318_features.py # Feature extraction pipeline (318+ signs)
│   ├── train_advanced_lstm.py  # Deep Learning LSTM training pipeline
│   ├── train_custom_model.py   # Basic model training (Scikit-Learn fallback)
│   ├── live_translator.py      # Standalone OpenCV-based translator
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Premium web dashboard
│   ├── style.css               # Ultra dark-mode glassmorphism UI
│   └── app.js                  # Frontend logic (skeleton, sparkline, modals, voice)
├── advanced_action_model.keras # Trained LSTM model (318+ signs)
├── label_map.pkl               # Sign label encoder
├── scaler.pkl                  # Feature scaler
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam
- Google Gemini API Key *(optional, for AI sentence building)*

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ISL-Translator.git
cd ISL-Translator
pip install -r backend/requirements.txt
```

### 2. Run the Application

```bash
cd backend
python api_server.py
```

Open **http://localhost:8000** in your browser and click **Start Translation**!

### 3. (Optional) Enable Gemini AI for Smart NLP

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Mac / Linux
export GEMINI_API_KEY=your_api_key_here
```

Get a free API key at [https://ai.google.dev](https://ai.google.dev)

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|:---:|:-------|
| `S` | Start Translation |
| `X` | Stop Translation |
| `C` | Clear Signs |
| `T` | AI Translate |
| `Space` | Speak Aloud |
| `M` | Toggle Mirror |
| `G` | Open Sign Gallery |
| `A` | Open Analytics |
| `Esc` | Close Modals |

---

## 🧠 Model Architecture

```
Input Layer      →  258 keypoints × 30 frames  =  7,740 features / sequence
                              ↓
LSTM Layer 1     →  128 units  +  Dropout(0.3)
                              ↓
LSTM Layer 2     →   64 units  +  Dropout(0.3)
                              ↓
Dense Layer      →   64 → 32 units  (ReLU + BatchNormalization)
                              ↓
Output Layer     →  318+ classes   (Softmax)
                              ↓
Threshold        →  Confidence > 70%  +  Stability Window (7 frames)
```

**Training Details:**
- Framework: TensorFlow 2.x / Keras
- Optimizer: Adam with learning rate scheduling
- Loss: Categorical Crossentropy
- Regularization: Dropout + Early Stopping

---

## 📊 Supported Signs (318+)

The model recognizes 318+ ISL signs including:

**Greetings:** `namaste`, `dhanyavaad`, `maaf_karo`, `alvida`  
**Pronouns:** `main`, `tum`, `woh`, `hum`, `aap`  
**Common Words:** `paani`, `khaana`, `ghar`, `school`, `hospital`, `help`  
**Time:** `aaj`, `kal`, `subah`, `shaam`, `raat`, `abhi`  
**Actions:** `aao`, `jao`, `baitho`, `khao`, `piyo`, `suno`  
**And 290+ more...** *(browse all in the in-app Sign Gallery)*

---

## 🖥️ UI Features

- **🎨 Glassmorphism Dark Mode** — Space Grotesk font, animated background orbs, noise texture overlay
- **🦴 Live Hand Skeleton** — Real-time cyan/purple landmark visualization on video feed  
- **📈 Confidence Sparkline** — Gradient line chart of model confidence history
- **🔵 SVG Progress Ring** — Circular arc shows buffer (0→30 frames) filling
- **📚 Sign Gallery Modal** — Searchable grid of all 318+ supported signs
- **🧠 Model Info Modal** — Visual LSTM architecture breakdown
- **📊 Analytics Modal** — Session duration, signs/min, top sign, frequency bar chart
- **🎓 Onboarding Tour** — 5-step guided tour for first-time visitors
- **🎤 Voice Commands** — Web Speech API: "start", "stop", "translate", "speak"
- **⚡ FPS Counter** — Live frames-per-second display on camera
- **🔊 Sound FX** — Subtle detection chime using Web Audio API
- **✍️ Typewriter Effect** — AI translation types out letter by letter

---

## 👩‍💻 About This Project

Built as a **portfolio project** demonstrating end-to-end AI engineering expertise:

- 🔬 **Computer Vision** — MediaPipe landmark extraction, hand skeleton rendering
- 📊 **Deep Learning** — Custom LSTM architecture, training pipeline, inference optimization  
- 🌐 **Full-Stack Development** — FastAPI backend, WebSocket real-time communication, premium frontend
- 🤖 **Generative AI** — Prompt engineering with Google Gemini 2.0

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ · Deep Learning · MediaPipe · FastAPI · Gemini AI

⭐ **Star this repo if you found it useful!** ⭐

</div>
