"""
ISL Translator - FastAPI Backend Server
Serves the ML model via WebSocket for real-time browser-based translation.
Also provides REST endpoints for NLP and TTS.
"""

# ⚠️ MUST be set before importing mediapipe or protobuf — fixes MessageFactory error
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pickle
import os
import json
import base64
import asyncio
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import our custom engines
from nlp_engine import NLPEngine
from tts_engine import TTSEngine

# ============ CONFIG ============
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'action_model.pkl')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'label_map.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), '..', 'frontend')
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.90  # High threshold — avoids random noise predictions
STABILITY_WINDOW = 5         # Slightly faster locking

# Signs to BLACKLIST — these classes are noisy and cause false positives
BLACKLISTED_SIGNS = {'bird', 'crowd', 'wide'}

# ============ LOAD MODEL ============
print("🧠 Loading AI Model...")
model = None
is_deep_learning = False

# Try Loading Neural Network (LSTM) First
ADVANCED_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'advanced_action_model.keras')
if os.path.exists(ADVANCED_MODEL_PATH):
    try:
        # TF 2.16+ uses standalone keras; older TF uses tensorflow.keras
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            from keras.models import load_model
        model = load_model(ADVANCED_MODEL_PATH)
        is_deep_learning = True
        print("✅ Advanced Deep Learning (LSTM) Model loaded!")
    except Exception as e:
        print(f"⚠️ LSTM Model load failed: {e}")

# Fallback to Basic Scikit-Learn
if model is None:
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ Basic Scikit-Learn Model loaded!")
    except FileNotFoundError:
        print("❌ No model found! Train the model first.")

# Load scaler
scaler = None
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded!")
except FileNotFoundError:
    print("⚠️  No scaler found, predictions may be less accurate.")

# Load actions
try:
    with open(LABEL_PATH, 'rb') as f:
        label_data = pickle.load(f)
        actions = label_data['actions']
    print(f"✅ {len(actions)} signs loaded: {list(actions)}")
except FileNotFoundError:
    actions = np.array(['namaste', 'paani', 'help'])
    print("⚠️  Using fallback actions")

# ============ MEDIAPIPE ============
# ============ KEYPOINT UTILS ============
def extract_keypoints_from_list(data):
    """Convert flat keypoints list from frontend into numpy array."""
    return np.array(data, dtype=np.float32)

# ============ FASTAPI APP ============
app = FastAPI(title="ISL Translator API", version="1.0.0")

# CORS - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
nlp_engine = NLPEngine()
tts_engine = TTSEngine()

# Serve frontend
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "signs_count": len(actions),
        "signs": [str(a) for a in actions],
        "nlp_available": nlp_engine.client is not None,
        "tts_available": tts_engine.enabled,
    }

@app.get("/api/signs")
async def get_signs():
    """Get list of all supported ISL signs."""
    return {"signs": [str(a) for a in actions], "count": len(actions)}

@app.post("/api/nlp")
async def translate_nlp(data: dict):
    """Convert raw sign words to a natural sentence."""
    words = data.get("words", [])
    sentence = nlp_engine.build_sentence(words)
    return {"sentence": sentence, "words": words}

@app.post("/api/speak")
async def speak_text(data: dict):
    """Speak the given text via TTS."""
    text = data.get("text", "")
    tts_engine.speak(text, force=True)
    return {"status": "speaking", "text": text}


@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ISL translation.
    Receives base64-encoded video frames from the browser,
    processes them through MediaPipe + ML model,
    and returns predictions.
    """
    await websocket.accept()
    print("🔗 WebSocket client connected!")

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    predictions = deque(maxlen=20)
    sentence = []
    last_prediction = ""

    try:
        while True:
            # Receive frame data from browser
            data = await websocket.receive_text()
            msg = json.loads(data)

            # ── LANDMARKS (from browser-side MediaPipe JS) ──────────────
            if msg.get("type") == "landmarks":
                landmark_data = msg.get("data", [])
                hands_detected = msg.get("hands_detected", False)

                if len(landmark_data) != 258:
                    continue

                keypoints = extract_keypoints_from_list(landmark_data)
                sequence.append(keypoints)

                # Build base response
                response = {
                    "type": "prediction",
                    "hands_detected": hands_detected,
                    "buffer_progress": len(sequence),
                    "buffer_needed": SEQUENCE_LENGTH,
                    "landmarks": {},  # Frontend draws skeleton directly from MediaPipe JS
                }

                # Predict when buffer is full
                if len(sequence) == SEQUENCE_LENGTH and model is not None:
                    hands_visible = hands_detected
                    
                    seq_np = np.array(list(sequence))
                    
                    try:
                        if is_deep_learning:
                            seq_input = seq_np.reshape(1, SEQUENCE_LENGTH, -1)
                            probs = model.predict(seq_input, verbose=0)[0]
                        else:
                            seq_flat = seq_np.reshape(1, -1)
                            if scaler is not None:
                                seq_flat = scaler.transform(seq_flat)
                            probs = model.predict_proba(seq_flat)[0]
                            
                        prediction_idx = np.argmax(probs)
                        confidence = float(probs[prediction_idx])

                        # Top-3 alternatives — useful when primary prediction is uncertain
                        top3_idx = np.argsort(probs)[::-1][:3]
                        top3 = [
                            {"sign": str(actions[i]), "confidence": round(float(probs[i]) * 100, 1)}
                            for i in top3_idx
                        ]
                        
                        if not hands_visible:
                            confidence = 0.0
                            current_sign = "Waiting for hands..."
                            top3 = []
                        elif str(actions[prediction_idx]) in BLACKLISTED_SIGNS:
                            # Skip noisy/generic signs — don't append to predictions
                            current_sign = str(actions[prediction_idx])
                            confidence = 0.0  # Show but don't commit to sentence
                        else:
                            predictions.append(prediction_idx)
                            current_sign = str(actions[prediction_idx])
                        
                        response["current_sign"] = current_sign
                        response["confidence"] = confidence
                        response["top3"] = top3

                        # Stability check
                        if hands_visible and len(predictions) >= STABILITY_WINDOW:
                            recent = list(predictions)[-STABILITY_WINDOW:]
                            if len(set(recent)) == 1 and confidence > CONFIDENCE_THRESHOLD:
                                if current_sign != last_prediction:
                                    sentence.append(current_sign)
                                    last_prediction = current_sign

                                    # Keep sentence manageable
                                    if len(sentence) > 10:
                                        sentence = sentence[-10:]

                        response["sentence"] = sentence
                        response["sentence_text"] = " ".join(sentence).upper()

                    except Exception as e:
                        response["error"] = str(e)

                await websocket.send_text(json.dumps(response))

            elif msg.get("type") == "clear":
                sentence = []
                last_prediction = ""
                predictions.clear()
                await websocket.send_text(json.dumps({
                    "type": "cleared",
                    "sentence": [],
                    "sentence_text": ""
                }))

            elif msg.get("type") == "speak":
                text = msg.get("text", "")
                tts_engine.speak(text, force=True)
                await websocket.send_text(json.dumps({"type": "spoken", "text": text}))

            elif msg.get("type") == "nlp":
                words = msg.get("words", [])
                sentence_text = nlp_engine.build_sentence(words)
                await websocket.send_text(json.dumps({
                    "type": "nlp_result",
                    "raw_words": words,
                    "sentence": sentence_text
                }))

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 WebSocket session ended")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting ISL Translator Server...")
    print("📍 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
