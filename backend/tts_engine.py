"""
Text-to-Speech Engine - Speaks out the translated ISL sentences via audio.
Uses browser-side Web Speech API as primary, with pyttsx3 as server-side fallback.
"""

import threading
import queue

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSEngine:
    """Text-to-Speech engine with lazy initialization to avoid blocking server startup."""

    def __init__(self, rate=160, volume=0.9):
        self.enabled = TTS_AVAILABLE
        self.speech_queue = queue.Queue()
        self.last_spoken = ""
        self._running = False
        self._initialized = False
        self._rate = rate
        self._volume = volume
        
        if self.enabled:
            print("✅ Text-to-Speech Engine ready (lazy init).")
        else:
            print("ℹ️  pyttsx3 not available. Using browser TTS fallback.")

    def _lazy_init(self):
        """Initialize pyttsx3 engine on first speak call (avoids blocking startup)."""
        if self._initialized:
            return
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self._rate)
            self.engine.setProperty('volume', self._volume)

            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            self._initialized = True
        except Exception as e:
            print(f"⚠️  TTS init failed: {e}")
            self.enabled = False

    def speak(self, text: str, force=False):
        """
        Queue text to be spoken.
        
        Args:
            text: The sentence to speak aloud.
            force: If True, speak even if same text was spoken before.
        """
        if not self.enabled or not text:
            return

        if not force and text == self.last_spoken:
            return

        # Lazy init on first speak
        if not self._initialized:
            self._lazy_init()
            if not self.enabled:
                return

        self.last_spoken = text
        self.speech_queue.put(text)

    def _worker(self):
        """Background thread that processes the speech queue."""
        while self._running:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️  TTS speak error: {e}")
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        """Gracefully shut down the TTS engine."""
        self._running = False
        if self._initialized:
            try:
                self.engine.stop()
            except:
                pass

    def reset(self):
        """Reset the last spoken text."""
        self.last_spoken = ""
