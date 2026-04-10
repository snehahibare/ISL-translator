---
title: ISL Translator
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# ISL Translator — Indian Sign Language AI

An advanced full-stack application that translates Indian Sign Language (ISL) into text and speech in real-time. This system leverages deep learning (LSTM) and computer vision (MediaPipe) to provide an accessible communication bridge for the Deaf and Hard of Hearing community.

### 🔗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Snehahibare/ISL-translator)

---

## 🚀 Key Features

*   **Real-Time Gesture Recognition:** Translates ISL signs instantly via webcam with high-precision hand tracking.
*   **Deep Learning Engine:** Powered by a Bidirectional LSTM neural network trained on over 318+ signs.
*   **Intelligent Sentence Building:** Uses NLP (Google Gemini) to transform raw sign words into grammatically correct English sentences.
*   **Multimodal Output:** Provides both real-time text display and Text-to-Speech (TTS) audio feedback.
*   **Interactive Analytics:** Track session performance including signs recognized, confidence stability, and session duration.
*   **Cross-Platform UI:** Modern, responsive dark-mode interface built with glassmorphism design principles.

---

## 🛠️ Tech Stack

*   **Computer Vision:** MediaPipe Holistic (for 258 body/hand landmarks).
*   **Deep Learning:** TensorFlow / Keras (Bidirectional LSTM).
*   **Backend:** FastAPI / Uvicorn (WebSocket-based real-time communication).
*   **Generative AI:** Google Gemini Pro (Natural Language Processing).
*   **Frontend:** Vanilla JavaScript, HTML5 Canvas, CSS3.
*   **Deployment:** Docker, Hugging Face Spaces.

---

## 🔮 How it Works

1.  **Landmark Extraction:** The browser captures video and uses MediaPipe JS to extract hand and body landmarks.
2.  **Sequence Processing:** Keypoints are streamed via WebSocket to the FastAPI backend, where they are collected into 30-frame sequences.
3.  **Inference:** The LSTM model analyzes the temporal patterns of the landmarks to predict the sign.
4.  **NLP Refinement:** Accumulated signs are sent to the Gemini AI engine to build a natural sentence.
5.  **Audio Synthesis:** The final translation is converted to speech via the Web Speech API.

---

## 📁 Project Overview

```text
├── backend/
│   ├── api_server.py           # FastAPI server & WebSocket handler
│   ├── nlp_engine.py           # Gemini AI integration
│   ├── tts_engine.py           # Text-to-Speech logic
│   └── train_advanced_lstm.py  # Model training architecture
├── frontend/
│   ├── index.html              # Core UI structure
│   ├── app.js                  # Real-time state & canvas management
│   └── style.css               # Styling and animations
└── Dockerfile                  # Containerization for deployment
```

---

## ⚙️ Local Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/snehahibare/ISL-translator.git
    cd ISL-translator
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **Run the Server:**
    ```bash
    python backend/api_server.py
    ```
    Access the application at `http://localhost:8000`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/snehahibare">Sneha Hibare</a>
</p>
