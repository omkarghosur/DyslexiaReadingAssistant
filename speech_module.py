import speech_recognition as sr
import streamlit as st
import random
import time
from gtts import gTTS
import tempfile
import os

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

def recognize_speech_unified(target_word, mode="advanced", slow_speed=False):
    """
    Enhanced speech recognition - User says full word letter by letter,
    then we recognize and compare all at once.
    """
    if not target_word:
        return None
    
    word_display_placeholder = st.empty()
    status_placeholder = st.empty()
    
    target_upper = target_word.upper().strip()
    
    # Initialize recognizer with optimized settings
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 2000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8
    
    def display_word_state(recognized_letters, comparing=False):
        """Display word with color-coded letters"""
        html_parts = []
        
        if comparing and recognized_letters:
            # Show comparison between target and recognized
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
                        # Incorrect - show what was said in red
                        html_parts.append(
                            f'<span style="color:#ff4b5c; font-size:70px; font-weight:bold; '
                            f'text-shadow:2px 2px 6px rgba(255,0,0,0.4); '
                            f'font-family: Comic Sans MS, Comfortaa, cursive;">{recognized_letters[i]}</span>'
                        )
                else:
                    # Missing letter
                    html_parts.append(
                        f'<span style="color:#ff4b5c; font-size:70px; font-weight:bold; '
                        f'text-shadow:2px 2px 6px rgba(255,0,0,0.4); '
                        f'font-family: Comic Sans MS, Comfortaa, cursive;">_</span>'
                    )
        else:
            # Show target word waiting for input
            for letter in target_upper:
                html_parts.append(
                    f'<span style="color:#2E86AB; font-size:70px; font-weight:bold; '
                    f'text-shadow:2px 2px 6px rgba(0,0,0,0.3); '
                    f'font-family: Comic Sans MS, Comfortaa, cursive;">{letter}</span>'
                )
        
        word_html = f"""
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
            status_placeholder.warning("🎤 **Calibrating microphone...** Please be quiet for 2 seconds...")
            display_word_state([])
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            if mode == "advanced":
                # Advanced mode - listen for full word spelled letter by letter
                status_placeholder.success(f"🎤 **Ready! Say '{target_word}' letter by letter**")
                status_placeholder.info(
                    f"💡 **Example:** For '{target_word}', say: "
                    f"'{' ... '.join(list(target_upper))}' (with pauses between letters)"
                )
                
                # Show target word
                display_word_state([])
                
                # Play example first
                play_sound(f"Say {target_word} letter by letter", slow=False)
                time.sleep(2)
                
                # Show ready prompt
                status_placeholder.info("🎤 **Listening now... Start speaking!**")
                time.sleep(1)
                
                try:
                    # Listen for the full spelling with longer timeout
                    # User will say: "C... A... T..." with pauses
                    audio = recognizer.listen(
                        source,
                        timeout=15,  # Long timeout for full word
                        phrase_time_limit=20  # Allow time for all letters
                    )
                    
                    status_placeholder.info("🔄 **Processing your speech...**")
                    
                    # Recognize what was said
                    spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                    
                    # Display what was heard
                    status_placeholder.success(f"✅ **Heard:** '{spoken_text}'")
                    
                    # Extract only letters from spoken text
                    spoken_letters = ''.join(c for c in spoken_text if c.isalpha())
                    
                    st.info(f"🔤 **Extracted letters:** {' '.join(list(spoken_letters))}")
                    
                    # Compare with target word letter by letter
                    recognized_letters = list(spoken_letters)
                    
                    # Pad with underscores if too short
                    while len(recognized_letters) < len(target_upper):
                        recognized_letters.append('_')
                    
                    # Trim if too long
                    recognized_letters = recognized_letters[:len(target_upper)]
                    
                    # Display comparison with animation
                    for i in range(len(target_upper) + 1):
                        display_word_state(recognized_letters[:i], comparing=True)
                        time.sleep(0.4)
                    
                    display_word_state(recognized_letters, comparing=True)
                    
                except sr.UnknownValueError:
                    status_placeholder.error("❌ Could not understand. Please speak clearly and try again.")
                    return None
                    
                except sr.WaitTimeoutError:
                    status_placeholder.error("⏱️ No speech detected. Please try again and start speaking sooner.")
                    return None
                    
                except sr.RequestError as e:
                    status_placeholder.error(f"❌ Speech service error: {str(e)}")
                    st.error("💡 **Check your internet connection**")
                    return None
                
            else:
                # Basic mode - recognize full word at once (not letter by letter)
                status_placeholder.success(f"🎤 **Say the word: '{target_word}' (as a complete word)**")
                display_word_state([])
                
                # Play the word first
                play_sound(target_word, slow=slow_speed)
                time.sleep(2)
                
                status_placeholder.info("🎤 **Listening now... Say the word!**")
                
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=6
                    )
                    
                    spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                    spoken_clean = ''.join(c for c in spoken_text if c.isalnum())
                    
                    status_placeholder.success(f"✅ **Heard:** '{spoken_text}'")
                    
                    # Compare letter by letter
                    recognized_letters = []
                    for i, target_letter in enumerate(target_upper):
                        if i < len(spoken_clean) and spoken_clean[i] == target_letter:
                            recognized_letters.append(target_letter)
                        elif target_letter in spoken_clean:
                            recognized_letters.append(target_letter)
                        else:
                            recognized_letters.append('_')
                    
                    # Show comparison with animation
                    for i in range(len(target_upper) + 1):
                        display_word_state(recognized_letters[:i], comparing=True)
                        time.sleep(0.3)
                    
                    display_word_state(recognized_letters, comparing=True)
                
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
            
            # Calculate and display accuracy
            correct_count = sum(1 for status in feedback.values() if status == "correct")
            accuracy = (correct_count / len(target_upper)) * 100
            
            if accuracy == 100:
                status_placeholder.success("🌟 **Perfect! 100% Correct!**")
                play_sound("Perfect! All letters correct!", slow=False)
            elif accuracy >= 70:
                status_placeholder.success(f"✅ **Great job! {accuracy:.0f}% Correct**")
                play_sound("Good job!", slow=False)
            else:
                status_placeholder.info(f"💪 **Keep trying! {accuracy:.0f}% Correct**")
                play_sound("Keep practicing!", slow=False)
            
            return feedback
            
    except OSError as e:
        status_placeholder.error(f"❌ Microphone error: {str(e)}")
        st.error("💡 **Troubleshooting:**")
        st.write("- Check microphone is connected and working")
        st.write("- Grant microphone permissions to your browser")
        st.write("- Close other apps using the microphone")
        st.write("- Try refreshing the page")
        return None
        
    except Exception as e:
        status_placeholder.error(f"❌ Unexpected error: {str(e)}")
        st.error("💡 **Try:** Refreshing the page or checking your microphone settings.")
        return None


# Backward compatibility
recognize_speech = recognize_speech_unified
recognize_speech_with_highlighting = recognize_speech_unified