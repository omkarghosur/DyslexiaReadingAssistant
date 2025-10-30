import speech_recognition as sr

def recognize_speech(expected_word):
    expected_word = expected_word.upper()
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"🎤 Please pronounce: {expected_word}")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5)
    
    try:
        spoken_text = recognizer.recognize_google(audio)
        print(f"🗣 You said: {spoken_text}")
        spoken_text_normalized = spoken_text.replace(" ", "").upper()  # remove spaces
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")
        return None

    result = {}
    for i, letter in enumerate(expected_word):
        if i < len(spoken_text_normalized) and letter == spoken_text_normalized[i]:
            result[letter] = "correct"
        else:
            result[letter] = "incorrect"

    return result

# Example usage
if __name__ == "__main__":
    feedback = recognize_speech("Bottle")
    if feedback:
        print("✅ Letter-by-letter feedback:", feedback)
