import speech_recognition as sr
import streamlit as st
import random
import time

def recognize_speech_unified(target_word, mode="advanced", slow_speed=False):
    """
    Enhanced speech recognition with letter-by-letter feedback.
    Now with proper sequential letter recognition.
    """
    if not target_word:
        return None
    
    word_display_placeholder = st.empty()
    status_placeholder = st.empty()
    
    target_upper = target_word.upper().strip()
    
    # Initialize recognizer with optimized settings
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.4
    
    def display_word_state(recognized_letters, current_index=-1, is_listening=False):
        """Display word with color-coded letters"""
        html_parts = []
        colors = ["#ff4b5c", "#f9ed69", "#6a2c70", "#1fab89", "#00bcd4", "#ff9800", "#cddc39"]
        
        for i, letter in enumerate(target_upper):
            if i < len(recognized_letters):
                if recognized_letters[i] == letter:
                    # Correct - green
                    html_parts.append(
                        f'<span style="color:#38ef7d; font-size:70px; font-weight:bold; '
                        f'text-shadow:2px 2px 6px rgba(0,255,0,0.4); '
                        f'font-family: Comic Sans MS, Comfortaa, cursive;">{letter}</span>'
                    )
                else:
                    # Incorrect - red
                    html_parts.append(
                        f'<span style="color:#ff4b5c; font-size:70px; font-weight:bold; '
                        f'text-shadow:2px 2px 6px rgba(255,0,0,0.4); '
                        f'font-family: Comic Sans MS, Comfortaa, cursive;">{letter}</span>'
                    )
            elif i == current_index and is_listening:
                # Currently listening - animated
                color = random.choice(colors)
                html_parts.append(
                    f'<span style="color:{color}; font-size:80px; font-weight:bold; '
                    f'text-shadow:3px 3px 10px rgba(0,0,0,0.5); '
                    f'font-family: Comic Sans MS, Comfortaa, cursive; '
                    f'animation: pulse 0.5s infinite;">{letter}</span>'
                )
            else:
                # Not yet recognized - gray
                html_parts.append(
                    f'<span style="color:#cccccc; font-size:64px; font-weight:bold; '
                    f'font-family: Comic Sans MS, Comfortaa, cursive;">{letter}</span>'
                )
        
        word_html = f"""
        <style>
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.15); }}
        }}
        </style>
        <div style="text-align:center; letter-spacing:15px; margin:30px 0; 
                    background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
                    border-radius:20px; padding:25px; border:2px solid rgba(255,255,255,0.2);">
            {''.join(html_parts)}
        </div>
        """
        word_display_placeholder.markdown(word_html, unsafe_allow_html=True)
    
    try:
        with sr.Microphone() as source:
            # Calibration
            status_placeholder.warning("🎤 **Calibrating...** Please be quiet for 2 seconds...")
            display_word_state([], is_listening=True)
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            recognized_letters = []
            
            if mode == "advanced":
                # Letter-by-letter recognition
                status_placeholder.success(f"🎤 **Let's spell '{target_word}' letter by letter!**")
                time.sleep(1)
                
                for i, target_letter in enumerate(target_upper):
                    display_word_state(recognized_letters, current_index=i, is_listening=True)
                    status_placeholder.info(f"🗣️ **Say the letter: '{target_letter}'** (Letter {i+1} of {len(target_upper)})")
                    
                    try:
                        # Listen for single letter
                        audio = recognizer.listen(
                            source,
                            timeout=5 if slow_speed else 3,
                            phrase_time_limit=2 if slow_speed else 1.5
                        )
                        
                        # Recognize what was said
                        spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                        spoken_clean = ''.join(c for c in spoken_text if c.isalnum())
                        
                        # Check if correct letter was said
                        if target_letter in spoken_clean:
                            recognized_letters.append(target_letter)
                            status_placeholder.success(f"✅ Correct! Heard: '{spoken_text}'")
                        else:
                            recognized_letters.append('_')
                            status_placeholder.error(f"❌ Expected '{target_letter}', heard: '{spoken_text}'")
                        
                        display_word_state(recognized_letters)
                        time.sleep(0.4)
                        
                    except sr.UnknownValueError:
                        recognized_letters.append('_')
                        display_word_state(recognized_letters)
                        status_placeholder.error(f"❌ Couldn't understand. Try saying '{target_letter}' more clearly.")
                        time.sleep(0.5)
                        
                    except sr.WaitTimeoutError:
                        recognized_letters.append('_')
                        display_word_state(recognized_letters)
                        status_placeholder.warning(f"⏱️ Timeout. Moving to next letter...")
                        time.sleep(0.5)
                        
                    except sr.RequestError as e:
                        status_placeholder.error(f"❌ Speech service error: {str(e)}")
                        return None
                
            else:
                # Basic mode - recognize full word at once
                status_placeholder.success(f"🎤 **Say the word: '{target_word}'**")
                display_word_state([], is_listening=True)
                
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=10 if slow_speed else 8,
                        phrase_time_limit=6 if slow_speed else 4
                    )
                    
                    spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                    spoken_clean = ''.join(c for c in spoken_text if c.isalnum())
                    
                    status_placeholder.success(f"✅ Heard: '{spoken_text}'")
                    
                    # Compare letter by letter
                    for i, target_letter in enumerate(target_upper):
                        if i < len(spoken_clean) and spoken_clean[i] == target_letter:
                            recognized_letters.append(target_letter)
                        elif target_letter in spoken_clean:
                            # Letter exists but wrong position
                            recognized_letters.append(target_letter)
                        else:
                            recognized_letters.append('_')
                        
                        display_word_state(recognized_letters)
                        time.sleep(0.2)
                
                except sr.UnknownValueError:
                    status_placeholder.error("❌ Could not understand. Please try again.")
                    return None
                except sr.WaitTimeoutError:
                    status_placeholder.error("⏱️ Timeout. Please try again.")
                    return None
                except sr.RequestError as e:
                    status_placeholder.error(f"❌ Speech service error: {str(e)}")
                    return None
            
            # Build feedback dictionary
            feedback = {}
            for i, letter in enumerate(target_upper):
                if i < len(recognized_letters):
                    feedback[letter] = "correct" if recognized_letters[i] == letter else "incorrect"
                else:
                    feedback[letter] = "incorrect"
            
            display_word_state(recognized_letters)
            status_placeholder.success("✅ **Recognition Complete!**")
            
            return feedback
            
    except OSError as e:
        status_placeholder.error(f"❌ Microphone error: {str(e)}")
        st.error("💡 **Check:** Microphone permissions, connection, and that no other app is using it.")
        return None
        
    except Exception as e:
        status_placeholder.error(f"❌ Unexpected error: {str(e)}")
        st.error("💡 **Try:** Refreshing the page or checking your microphone settings.")
        return None


# Backward compatibility
recognize_speech = recognize_speech_unified
recognize_speech_with_highlighting = recognize_speech_unified