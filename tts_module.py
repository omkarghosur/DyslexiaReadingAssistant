import pyttsx3

class DyslexiaTTS:
    def __init__(self, rate_letters=150, rate_word=160, volume=1.0):
        """
        Initialize default parameters for speech rates and volume.
        """
        self.rate_letters = rate_letters
        self.rate_word = rate_word
        self.volume = volume
        self.spoken_words = set()  # Track spoken words to avoid repetition

    def speak_text(self, word):
        """
        Pronounce a word letter by letter, then the full word immediately.
        Will speak each word only once.
        """
        word = word.strip()
        if not word or word in self.spoken_words:
            return  # Skip if empty or already spoken

        print(f"🔤 Speaking: {word}")
        self.spoken_words.add(word)  # Mark as spoken

        # Create a fresh engine instance
        engine = pyttsx3.init()
        engine.setProperty('volume', self.volume)

        # Letters spaced for clear pronunciation
        letters_spaced = " ".join(list(word.upper()))
        engine.setProperty('rate', self.rate_letters)
        engine.say(letters_spaced)

        # Full word immediately after
        engine.setProperty('rate', self.rate_word)
        engine.say(word)

        engine.runAndWait()
        engine.stop()
from tts_module import DyslexiaTTS

tts = DyslexiaTTS(rate_letters=150, rate_word=160)

words = ["hello" ]  # last "Bottle" won't repeat
for w in words:
    tts.speak_text(w)
