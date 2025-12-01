import speech_recognition as sr
import streamlit as st
import time
from gtts import gTTS
import tempfile
import os
import difflib

def play_sound(text, slow=False):
    """Helper function to play audio feedback"""
    try:
        tts = gTTS(text=text, lang='en', slow=slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            try:
                from playsound import playsound
                playsound(f.name)
            except:
                pass
            try:
                os.remove(f.name)
            except:
                pass
    except Exception as e:
        pass

def phonetic_match(spoken_letter, target_letter):
    """
    Check if spoken letter matches target letter.
    STRICT MODE: Only accepts the letter or its specific phonetic name.
    """
    spoken = spoken_letter.upper()
    target = target_letter.upper()
    
    if spoken == target: return True
    
    phonetic_groups = {
        'A': ['AY', 'EY'], 'B': ['BEE', 'BE'], 'C': ['SEE', 'SEA'],
        'D': ['DEE'], 'E': ['EE'], 'F': ['EFF', 'EF'],
        'G': ['GEE', 'JE'], 'H': ['AYCH', 'AITCH', 'HA'],
        'I': ['EYE', 'AYE'], 'J': ['JAY'], 'K': ['KAY', 'CAY'],
        'L': ['ELL', 'EL'], 'M': ['EM'], 'N': ['EN'],
        'O': ['OH', 'OWE'], 'P': ['PEE', 'PE'],
        'Q': ['CUE', 'KYU', 'QUE'], 'R': ['AR', 'ARE'],
        'S': ['ESS', 'ES'], 'T': ['TEE', 'TE'],
        'U': ['YOU', 'YUE'], 'V': ['VEE', 'VE'],
        'W': ['DOUBLE-U', 'DOUBLEU'], 'X': ['EX'],
        'Y': ['WHY', 'WYE'], 'Z': ['ZEE', 'ZED']
    }
    
    if target in phonetic_groups and spoken in phonetic_groups[target]:
        return True
    return False

def extract_letters_from_speech(spoken_text, target_length):
    """
    Extracts explicit letters from speech.
    """
    letter_names = {
        'AY': 'A', 'BEE': 'B', 'SEE': 'C', 'DEE': 'D', 'EE': 'E',
        'EFF': 'F', 'GEE': 'G', 'AYCH': 'H', 'EYE': 'I', 'JAY': 'J',
        'KAY': 'K', 'ELL': 'L', 'EM': 'M', 'EN': 'N', 'OH': 'O',
        'PEE': 'P', 'CUE': 'Q', 'AR': 'R', 'ESS': 'S', 'TEE': 'T',
        'YOU': 'U', 'VEE': 'V', 'DOUBLE': 'W', 'EX': 'X', 'WHY': 'Y',
        'ZEE': 'Z', 'ZED': 'Z', 'SPACE': ' '
    }
    
    words = spoken_text.upper().split()
    recognized_letters = []
    
    i = 0
    while i < len(words):
        word = words[i]
        
        # Handle "DOUBLE U"
        if word == 'DOUBLE' and i + 1 < len(words) and words[i+1] in ['U', 'YOU']:
            recognized_letters.append('W')
            i += 2; continue
            
        if word in letter_names:
            recognized_letters.append(letter_names[word])
        elif len(word) == 1 and word.isalpha():
            recognized_letters.append(word)
        elif word == 'SPACE':
            recognized_letters.append(' ')
        elif word.isalpha():
            recognized_letters.append(word[0])
            
        i += 1
    return recognized_letters

def recognize_speech_unified(target_word, mode="spelling", slow_speed=False):
    """
    Unified function for both Spelling and Pronunciation.
    """
    if not target_word: return None
    
    word_display_placeholder = st.empty()
    status_placeholder = st.empty()
    target_upper = target_word.upper().strip()
    
    recognizer = sr.Recognizer()
    
    def display_word_state(match_results, is_final=False):
        html_parts = []
        if is_final:
            for i, letter in enumerate(target_upper):
                is_correct = False
                if i < len(match_results) and match_results[i] is True:
                    is_correct = True
                
                color = "#38ef7d" if is_correct else "#ff4b5c"
                display_char = letter if letter != ' ' else '&nbsp;&nbsp;'
                html_parts.append(f'<span style="color:{color}; font-size:70px; font-weight:bold; text-shadow:2px 2px 6px rgba(0,0,0,0.2); margin: 0 5px;">{display_char}</span>')
        else:
            for letter in target_upper:
                display_char = letter if letter != ' ' else '&nbsp;&nbsp;'
                html_parts.append(f'<span style="color:#2E86AB; font-size:70px; font-weight:bold; margin: 0 5px;">{display_char}</span>')

        word_html = f"""
        <div style="text-align:center; margin:30px 0; background: rgba(255,255,255,0.1); 
                    border-radius:20px; padding:25px; border:2px solid rgba(255,255,255,0.2);">
            {''.join(html_parts)}
        </div>"""
        word_display_placeholder.markdown(word_html, unsafe_allow_html=True)

    try:
        with sr.Microphone() as source:
            status_placeholder.warning("🤫 Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            display_word_state([], is_final=False)
            
            if mode == "spelling":
                status_placeholder.info(f"🎤 **SPELLING**: Say '{' ... '.join(list(target_upper))}'")
                play_sound("Spell the word letter by letter", slow=False)
            else:
                status_placeholder.info(f"🎤 **SPEAKING**: Say '{target_word}'")
                play_sound(f"Say the word {target_word}", slow=slow_speed)
            
            time.sleep(0.5)
            status_placeholder.success("🎤 **LISTENING... GO!**")
            
            try:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=15 if mode == "spelling" else 5)
                status_placeholder.info("🔄 **Processing...**")
                
                spoken_text = ""
                try:
                    spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                    st.toast(f"Heard: {spoken_text}")
                except sr.UnknownValueError:
                    spoken_text = "" 
                
                feedback = {}
                match_results = [False] * len(target_upper)
                
                if mode == "spelling":
                    spoken_letters = extract_letters_from_speech(spoken_text, len(target_upper))
                    spoken_idx = 0
                    
                    # --- FIXED LOGIC FOR SPACES ---
                    for i, target_char in enumerate(target_upper):
                        if target_char == ' ':
                            # If user explicitly said "Space", consume it
                            if spoken_idx < len(spoken_letters) and spoken_letters[spoken_idx] == ' ':
                                match_results[i] = True
                                spoken_idx += 1
                            else:
                                # User didn't say space? Auto-mark Correct! (Don't break flow)
                                match_results[i] = True
                        else:
                            # Normal Letter Matching
                            if spoken_idx < len(spoken_letters):
                                if spoken_letters[spoken_idx] == ' ': # Skip accidental extra space in speech
                                    spoken_idx += 1
                                    
                                if spoken_idx < len(spoken_letters):
                                    sl = spoken_letters[spoken_idx]
                                    if sl == target_char or phonetic_match(sl, target_char):
                                        match_results[i] = True
                                    spoken_idx += 1
                        
                        feedback[target_char] = "correct" if match_results[i] else "incorrect"

                else: 
                    # Pronunciation Mode
                    spoken_clean = ''.join(c for c in spoken_text if c.isalnum())
                    matcher = difflib.SequenceMatcher(None, target_upper, spoken_clean)
                    for match_id, (i, j, n) in enumerate(matcher.get_matching_blocks()):
                        for k in range(n):
                            if i + k < len(match_results):
                                match_results[i + k] = True

                    for i, letter in enumerate(target_upper):
                        feedback[letter] = "correct" if match_results[i] else "incorrect"
                    
                    if not spoken_clean: st.warning("⚠️ Didn't catch that.")
                    elif not all(match_results): st.warning(f"👂 You said: **{spoken_text}**")

                display_word_state(match_results, is_final=True)
                
                correct_count = sum(match_results)
                if correct_count == len(target_upper) and len(target_upper) > 0:
                    status_placeholder.success("🌟 Perfect Match!")
                elif correct_count > 0:
                    status_placeholder.warning(f"👍 Partial Match! ({correct_count}/{len(target_upper)})")
                else:
                    status_placeholder.error("❌ Not quite. Try again!")
                
                return feedback

            except sr.WaitTimeoutError:
                status_placeholder.error("⏱️ Timeout. Please speak louder.")
                return None
            except sr.RequestError:
                status_placeholder.error("❌ Internet/API Error.")
                return None
                
    except Exception as e:
        st.error(f"Error: {e}")
        return None