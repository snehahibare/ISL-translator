"""
NLP Engine - Converts raw ISL sign words into natural, grammatical sentences.
Uses Google Gemini API for intelligent sentence construction.
"""

import os

# Try to import Google GenAI (new SDK)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-genai not installed. Install with: pip install google-genai")


class NLPEngine:
    """Converts a sequence of ISL sign words into a natural English sentence using Gemini AI."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None

        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("✅ Gemini NLP Engine initialized!")
            except Exception as e:
                print(f"⚠️  Gemini init failed: {e}")
                self.client = None
        else:
            if not self.api_key:
                print("⚠️  No GEMINI_API_KEY found. Set it as environment variable or pass to constructor.")
            print("ℹ️  NLP Engine running in OFFLINE mode (basic word joining).")

    def build_sentence(self, words: list) -> str:
        """
        Takes a list of ISL sign words and returns a grammatically correct English sentence.

        Args:
            words: List of detected sign words, e.g. ["main", "paani", "help"]

        Returns:
            A natural English sentence, e.g. "I need help getting some water."
        """
        if not words:
            return ""

        clean_words = [w.strip().replace("_", " ") for w in words if w.strip()]
        if not clean_words:
            return ""

        if self.client:
            return self._gemini_translate(clean_words)
        else:
            return self._offline_translate(clean_words)

    def _gemini_translate(self, words: list) -> str:
        """Use Gemini AI to construct a natural sentence from ISL words."""
        prompt = f"""You are an Indian Sign Language (ISL) interpreter AI. 
You receive a sequence of Hindi/ISL sign words detected from a deaf person's hand gestures.
Your job is to form a natural, grammatically correct English sentence from these words.

Rules:
- The words are in the ORDER they were signed
- Some words are Hindi (like "main" = "I", "paani" = "water", "khaana" = "food", "ghar" = "home")
- Keep the sentence short and natural (1 line max)
- Do NOT add extra context that wasn't in the signs
- Return ONLY the final English sentence, nothing else

ISL Words detected: {', '.join(words)}

English sentence:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            sentence = response.text.strip().strip('"').strip("'")
            sentence = sentence.replace("*", "").replace("#", "").strip()
            return sentence
        except Exception as e:
            print(f"⚠️  Gemini API error: {e}")
            return self._offline_translate(words)

    def _offline_translate(self, words: list) -> str:
        """Basic offline translation using a Hindi-to-English dictionary."""
        hindi_to_english = {
            "namaste": "Hello",
            "main": "I",
            "tum": "you",
            "woh": "they",
            "paani": "water",
            "khaana": "food",
            "ghar": "home",
            "school": "school",
            "hospital": "hospital",
            "help": "help",
            "haan": "yes",
            "nahi": "no",
            "dhanyavaad": "thank you",
            "maaf karo": "sorry",
            "aaj": "today",
            "kal": "tomorrow",
            "subah": "morning",
            "shaam": "evening",
            "theek hai": "okay",
            "emergency": "emergency",
        }

        translated = []
        for word in words:
            eng = hindi_to_english.get(word.lower(), word.capitalize())
            translated.append(eng)

        sentence = " ".join(translated)
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence


# Quick test
if __name__ == "__main__":
    engine = NLPEngine()

    test_cases = [
        ["namaste", "main", "paani"],
        ["help", "emergency", "hospital"],
        ["main", "ghar", "kal"],
        ["dhanyavaad"],
    ]

    print("\n--- NLP Engine Test ---")
    for words in test_cases:
        result = engine.build_sentence(words)
        print(f"  Signs: {words}")
        print(f"  → {result}\n")
