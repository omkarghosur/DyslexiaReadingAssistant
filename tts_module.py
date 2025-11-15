from gtts import gTTS
import os
from playsound import playsound
import tempfile
import time

class DyslexiaTTS:
    def __init__(self, lang='en', slow_letters=True, slow_word=False):
        """
        Initialize TTS settings.
        :param lang: Language code (default 'en' for English)
        :param slow_letters: Speak letters slowly for clarity
        :param slow_word: Speak word slowly or normally
        """
        self.lang = lang
        self.slow_letters = slow_letters
        self.slow_word = slow_word
        self.spoken_words = set()  # To avoid repetition

    def speak_text(self, word):
        """
        Spell and pronounce a given word or phrase.
        Handles multi-word phrases properly.
        Speaks letter-by-letter first, then the whole word/phrase.
        """
        word = word.strip()
        if not word or word.lower() in self.spoken_words:
            return

        print(f"🔤 Speaking: {word}")
        self.spoken_words.add(word.lower())

        # Create temporary files for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f_letters, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f_word:

            try:
                # 1️⃣ Spell letter by letter (skip spaces)
                # Extract only alphanumeric characters for spelling
                letters_only = [char for char in word.upper() if char.isalnum()]
                letters_spaced = " ".join(letters_only)
                
                print(f"  Letters: {letters_spaced}")
                
                tts_letters = gTTS(text=letters_spaced, lang=self.lang, slow=self.slow_letters)
                tts_letters.save(f_letters.name)
                playsound(f_letters.name)

                time.sleep(0.5)  # Pause between spelling and full word

                # 2️⃣ Speak full word/phrase
                print(f"  Full word: {word}")
                
                tts_word = gTTS(text=word, lang=self.lang, slow=self.slow_word)
                tts_word.save(f_word.name)
                playsound(f_word.name)

            except Exception as e:
                print(f"❌ TTS Error: {e}")
            finally:
                # Cleanup temp files
                try:
                    os.remove(f_letters.name)
                except:
                    pass
                try:
                    os.remove(f_word.name)
                except:
                    pass