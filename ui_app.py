import streamlit as st
import cv2
import google.generativeai as genai
import os
import base64
import requests
import time
import random
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gtts import gTTS
import tempfile
from playsound import playsound
import speech_recognition as sr

# Import our custom modules
from tts_module import DyslexiaTTS
from speech_module import recognize_speech

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.warning("⚠️ GOOGLE_API_KEY not found. Object detection and AI feedback are disabled.")
    model = None
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
    except Exception as _e:
        model = None
        st.warning("⚠️ Gemini model could not be initialized. Some features are disabled.")

# ========================================
# 🎨 SYNCHRONIZED LETTER HIGHLIGHTING + TTS
# ========================================
def spell_word_with_highlighting(word, slow_letters=True, slow_word=False):
    """Spell word letter-by-letter with synchronized visual highlighting and audio."""
    placeholder = st.empty()
    colors = ["#ff4b5c", "#f9ed69", "#6a2c70", "#1fab89", "#00bcd4", "#ff9800", "#cddc39"]
    
    word_upper = word.upper()
    
    # Step 1: Spell each letter with synchronized audio + visual
    for i, letter in enumerate(word_upper):
        # Choose random color for this letter
        color = random.choice(colors)
        
        # Build HTML showing all letters, with current one highlighted
        html_parts = []
        for j, char in enumerate(word_upper):
            if j == i:
                # Current letter - highlighted in color
                html_parts.append(
                    f"<span style='color:{color}; font-size:80px; font-weight:bold; "
                    f"text-shadow:3px 3px 8px rgba(0,0,0,0.4); font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{char}</span>"
                )
            else:
                # Other letters - dimmed
                html_parts.append(
                    f"<span style='color:#cccccc; font-size:64px; font-weight:bold; font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{char}</span>"
                )
        
        # Display the word with current letter highlighted
        full_html = f"""
        <div style='text-align:center; letter-spacing:12px; margin:30px 0;'>
            {''.join(html_parts)}
        </div>
        """
        placeholder.markdown(full_html, unsafe_allow_html=True)
        
        # Play audio for this letter (synchronized!)
        try:
            letter_tts = gTTS(text=letter, lang='en', slow=slow_letters)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                letter_tts.save(f.name)
                playsound(f.name)
                os.remove(f.name)
        except Exception as e:
            st.error(f"Audio error for letter {letter}: {e}")
        
        # Small pause between letters
        time.sleep(0.2)
    
    # Step 2: Show full word and speak it
    time.sleep(0.3)
    
    final_html = f"""
    <div style='text-align:center; margin:30px 0;'>
        <div style='font-size:72px; font-weight:bold; color:#2E86AB; 
                    text-shadow:2px 2px 6px rgba(0,0,0,0.3); letter-spacing:8px;
                    font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>
            {word_upper}
        </div>
        <div style='font-size:24px; color:#666; margin-top:15px; font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>
            Now say it together! 👇
        </div>
    </div>
    """
    placeholder.markdown(final_html, unsafe_allow_html=True)
    
    # Speak full word
    try:
        word_tts = gTTS(text=word, lang='en', slow=slow_word)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            word_tts.save(f.name)
            playsound(f.name)
            os.remove(f.name)
    except Exception as e:
        st.error(f"Audio error for full word: {e}")

# ========================================
# 🎤 UNIFIED SPEECH RECOGNITION SYSTEM
# ========================================
def recognize_speech_unified(target_word, mode="advanced", slow_speed=False):
    """
    Unified speech recognition system that works for both basic and advanced modes.
    
    Args:
        target_word: The target word to practice
        mode: "advanced" (with real-time highlighting) or "basic" (simple feedback)
        slow_speed: If True, gives more time for pronunciation
    
    Returns:
        Dictionary with letter-by-letter feedback
    """
    if not target_word:
        return None
    
    # Create placeholders for dynamic updates
    word_display_placeholder = st.empty()
    status_placeholder = st.empty()
    
    target_upper = target_word.upper()
    
    # Display function for word state
    def display_word_state(recognized_letters, current_index=-1, is_listening=False):
        """Display word with color-coded letters based on recognition state"""
        html_parts = []
        colors = ["#ff4b5c", "#f9ed69", "#6a2c70", "#1fab89", "#00bcd4", "#ff9800", "#cddc39"]
        
        for i, letter in enumerate(target_upper):
            if i < len(recognized_letters):
                if recognized_letters[i] == letter:
                    # Correct letter - green gradient
                    html_parts.append(
                        f"<span style='font-size:70px; font-weight:bold; "
                        f"text-shadow:2px 2px 6px rgba(0,255,0,0.4); "
                        f"font-family: \"Comic Sans MS\", \"Comfortaa\", cursive; "
                        f"background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%); "
                        f"-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>{letter}</span>"
                    )
                else:
                    # Incorrect letter - red
                    html_parts.append(
                        f"<span style='color:#ff4b5c; font-size:70px; font-weight:bold; "
                        f"text-shadow:2px 2px 6px rgba(255,0,0,0.4); "
                        f"font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{letter}</span>"
                    )
            elif i == current_index and mode == "advanced":
                # Currently being recognized - animated color
                color = random.choice(colors)
                html_parts.append(
                    f"<span style='color:{color}; font-size:80px; font-weight:bold; "
                    f"text-shadow:3px 3px 10px rgba(0,0,0,0.5); "
                    f"font-family: \"Comic Sans MS\", \"Comfortaa\", cursive; "
                    f"animation: pulse 0.5s infinite;'>{letter}</span>"
                )
            else:
                # Not yet recognized - gray
                gray_color = "#ffffff" if is_listening else "#cccccc"
                html_parts.append(
                    f"<span style='color:{gray_color}; font-size:64px; font-weight:bold; "
                    f"font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{letter}</span>"
                )
        
        word_html = f"""
        <style>
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        </style>
        <div style='text-align:center; letter-spacing:15px; margin:40px 0; padding:20px;
                    background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
                    border-radius:20px; border:2px solid rgba(255,255,255,0.2);'>
            {''.join(html_parts)}
        </div>
        """
        word_display_placeholder.markdown(word_html, unsafe_allow_html=True)
    
    # Initial display - show full word
    display_word_state([])
    status_placeholder.info("🎤 **Get Ready!** Speak the word clearly when you see 'Listening...'")
    time.sleep(1.5)
    
    # Initialize speech recognizer with optimized settings
    recognizer = sr.Recognizer()
    
    # Optimized recognizer settings for better accuracy
    recognizer.energy_threshold = 4000  # Higher threshold to reduce background noise
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Wait 0.8s of silence before considering phrase complete
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5
    
    recognized_letters = []
    
    try:
        # Try to access microphone with error handling
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                status_placeholder.warning("🎤 **Calibrating...** Please wait (be quiet for 2 seconds)...")
                display_word_state([], is_listening=True)
                recognizer.adjust_for_ambient_noise(source, duration=2)
                
                # Ready to listen
                status_placeholder.success(f"🎤 **Listening...** Say '{target_word}' clearly now!")
                display_word_state([], is_listening=True)
                
                # Listen with longer timeout for better results
                timeout_duration = 15 if slow_speed else 10
                phrase_limit = 8 if slow_speed else 6
                
                audio = recognizer.listen(source, timeout=timeout_duration, phrase_time_limit=phrase_limit)
                
                # Processing audio
                status_placeholder.info("🔄 **Processing...** Analyzing your pronunciation...")
                
                # Try Google Speech Recognition
                try:
                    spoken_text = recognizer.recognize_google(audio, language='en-US').upper()
                    status_placeholder.success(f"✅ **Heard:** '{spoken_text}'")
                    
                    # Remove spaces and special characters for comparison
                    spoken_clean = ''.join(c for c in spoken_text if c.isalnum())
                    
                    # Analyze letter by letter with animation
                    for i, letter in enumerate(target_upper):
                        display_word_state(recognized_letters, current_index=i, is_listening=True)
                        status_placeholder.info(f"🎤 Say the letter: **{letter}**")
                             
                        try:   
                            partial_audio = recognizer.listen(source, timeout=3, phrase_time_limit=1.5) 
                            partial_text = recognizer.recognize_google(partial_audio, language='en-US').upper()

                            if letter in partial_text:
                                recognized_letters.append(letter)
                                display_word_state(recognized_letters)
                                st.success(f"✅ Heard: {letter}")
                            else:
                                recognized_letters.append('_')
                                display_word_state(recognized_letters)
                                st.warning(f"⚠️ Letter '{letter}' not detected")

                        except sr.UnknownValueError:
                            recognized_letters.append('_')
                            display_word_state(recognized_letters)
                            st.warning(f"❌ Couldn't understand '{letter}' — try again")

                        except sr.WaitTimeoutError:
                            recognized_letters.append('_')
                            display_word_state(recognized_letters)
                            st.warning(f"⏱️ No sound detected for '{letter}'")
                        # ✅ Final display after loop
                        display_word_state(recognized_letters)        
                        
                        # Compare letters with multiple strategies
                        letter_found = False
                        
                        # Strategy 1: Position-based matching
                        if i < len(spoken_clean) and spoken_clean[i] == target_letter:
                            recognized_letters.append(target_letter)
                            letter_found = True
                        # Strategy 2: Contains matching (for out-of-order speech)
                        elif target_letter in spoken_clean:
                            # Count occurrences to avoid reusing same letter
                            target_count = target_upper[:i+1].count(target_letter)
                            spoken_count = spoken_clean.count(target_letter)
                            if spoken_count >= target_count:
                                recognized_letters.append(target_letter)
                                letter_found = True
                        
                        if not letter_found:
                            recognized_letters.append('_')
                        
                        # Update display with recognition result
                        if mode == "advanced":
                            display_word_state(recognized_letters)
                            time.sleep(0.2)
                    
                    # Final result display
                    display_word_state(recognized_letters)
                    
                    # Build detailed feedback dictionary
                    feedback = {}
                    for i, letter in enumerate(target_upper):
                        if i < len(recognized_letters) and recognized_letters[i] == letter:
                            feedback[letter] = "correct"
                        else:
                            feedback[letter] = "incorrect"
                    
                    status_placeholder.success("✅ **Analysis Complete!** Check your results below.")
                    return feedback
                    
                except sr.UnknownValueError:
                    status_placeholder.error("❌ **Could not understand audio.**")
                    st.warning("💡 **Tips to improve:**")
                    st.write("- Speak louder and more clearly")
                    st.write("- Move closer to your microphone")
                    st.write("- Reduce background noise")
                    st.write("- Spell out the word letter by letter: P-H-O-N-E")
                    return None
                    
                except sr.RequestError as e:
                    status_placeholder.error(f"❌ **Speech service error:** {str(e)}")
                    st.error("💡 **Connection issue detected:**")
                    st.write("- Check your internet connection")
                    st.write("- Try again in a few moments")
                    st.write("- Use the 'Basic Recognition' button as fallback")
                    return None
        
        except OSError as e:
            status_placeholder.error(f"❌ **Microphone access error:** {str(e)}")
            st.error("💡 **Microphone troubleshooting:**")
            st.write("- Ensure microphone is connected")
            st.write("- Check microphone permissions in browser settings")
            st.write("- Try refreshing the page")
            st.write("- Make sure no other app is using the microphone")
            return None
                
    except sr.WaitTimeoutError:
        status_placeholder.error("⏱️ **Timeout!** No speech detected.")
        st.warning("💡 **Please:**")
        st.write("- Click the button again")
        st.write("- Speak immediately when you see 'Listening...'")
        st.write("- Make sure your microphone is working")
        return None
        
    except Exception as e:
        status_placeholder.error(f"❌ **Unexpected error:** {str(e)}")
        st.error("💡 **Something went wrong. Please:**")
        st.write("- Try refreshing the page")
        st.write("- Check your microphone settings")
        st.write("- Use manual input as a fallback")
        return None

# Object Detection Functions

# ✅ Fix: Make old function name work
recognize_speech_with_highlighting = recognize_speech_unified



def get_text_from_image_gemini(frame):
    """Extracts text from image using Gemini API."""
    if not api_key:
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Extract the text from this image. Be precise and only return the text."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
                ]
            }
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        candidates = result.get('candidates', [])
        if candidates and 'content' in candidates[0]:
            parts = candidates[0]['content'].get('parts', [])
            text = parts[0].get('text', '') if parts else ''
            return text.strip()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Gemini API Error: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Gemini Response Parsing Error: {e}")
        return None

def get_object_detection_gemini(frame):
    """Detects objects in image using Gemini API with retry logic."""
    if not api_key:
        return None
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Resize frame for better processing
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Look at this image carefully. Identify the main object that a person is showing or holding. Ignore human body parts, background items, and multiple objects. Return only the single most prominent object name in English, like 'Apple', 'Book', 'Pen', 'Phone', etc. Be specific and concise."},
                            {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
                        ]
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            
            session = requests.Session()
            session.verify = True
            
            response = session.post(api_url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            
            candidates = result.get('candidates', [])
            if candidates and 'content' in candidates[0]:
                parts = candidates[0]['content'].get('parts', [])
                text = parts[0].get('text', '') if parts else ''
                detected_text = text.strip()
                
                prefixes_to_remove = ["The object is", "I can see", "This is", "The main object is", "Object:"]
                for prefix in prefixes_to_remove:
                    if detected_text.lower().startswith(prefix.lower()):
                        detected_text = detected_text[len(prefix):].strip()
                
                detected_text = detected_text.split('.')[0].split(',')[0].strip()
                
                return detected_text if detected_text else None
            
            return None
            
        except requests.exceptions.SSLError as e:
            st.warning(f"🔄 SSL Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                st.error("❌ SSL connection failed after multiple attempts.")
                return None
                
        except requests.exceptions.RequestException as e:
            st.warning(f"🔄 Request Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                st.error(f"❌ API request failed: {e}")
                return None
                
        except (KeyError, IndexError) as e:
            st.error(f"❌ Gemini Response Parsing Error: {e}")
            return None
            
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return None
    
    return None

def get_pronunciation_feedback(word):
    """Gets letter-by-letter pronunciation feedback from Gemini."""
    if not word:
        return "No text to analyze."
    if not api_key:
        return "Gemini feedback disabled (no API key)."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"A dyslexic student is learning to pronounce the word '{word}'. "
                            "Break the word into letters and explain how each letter sounds "
                            "in **simple English**, not phonetic symbols. "
                            "Example: For 'cat' → C says 'kuh', A says 'aah', T says 'tuh'. "
                            "Be short, friendly, and encouraging. "
                            "End with a full-word pronunciation like: 'Now say it together: cat!'."
                        )
                    }
                ]
            }
        ]
    }

    headers = {'Content-Type': 'application/json'}
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        result = response.json()

        candidates = result.get('candidates', [])
        if candidates and 'content' in candidates[0]:
            parts = candidates[0]['content'].get('parts', [])
            text = parts[0].get('text', '').strip() if parts else ''
            return text or "No feedback received."
        return "No feedback received."

    except requests.exceptions.Timeout:
        return "⚠️ Gemini took too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Feedback Error: {e}"
    except (KeyError, IndexError) as e:
        return f"Feedback Parsing Error: {e}"


# Streamlit UI
st.set_page_config(
    page_title="AI Reading Assistant",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS with consistent Comic Sans MS font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');

/* Global font family */
* {
    font-family: "Comic Sans MS", "Comfortaa", cursive !important;
}

/* Simplified animated background */
@keyframes gradient { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }
#root { background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe); background-size: 400% 400%; animation: gradient 15s ease infinite; min-height: 100vh; }

/* Glassmorphism card */
.main-container, .glass-card { background: rgba(255,255,255,0.15); backdrop-filter: blur(16px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2); padding: 1.5rem; }

/* Buttons */
.stButton>button { background: linear-gradient(135deg,#667eea,#764ba2); border:none; border-radius:12px; color:#fff !important; font-weight:600; padding:0.5rem 1.25rem; box-shadow:0 4px 15px rgba(116,79,168,0.45); font-family: "Comic Sans MS", "Comfortaa", cursive !important; }
.stButton>button:hover { transform: translateY(-2px); background: linear-gradient(135deg,#764ba2,#f093fb); }

/* Typography */
h1,h2,h3,h4,h5,h6 { font-family: "Comic Sans MS", "Comfortaa", cursive !important; background: linear-gradient(135deg,#667eea 0%, #f093fb 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:2px; }

/* Letter feedback */
/* Letter feedback */
.letter-correct {
    background: linear-gradient(135deg,#38ef7d 0%,#11998e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 48px;
}

.letter-incorrect {
    background: linear-gradient(135deg,#ff4b5c 0%,#f9ed69 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 48px;
}


/* Inputs */
.stTextInput>div>div>input { background: rgba(255,255,255,0.2); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.3); border-radius:10px; color:#fff; font-family: "Comic Sans MS", "Comfortaa", cursive !important; }

/* Scrollbar */
::-webkit-scrollbar { width:10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_" + str(int(time.time()))
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + str(int(time.time()))
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""

# Initialize TTS engine
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = None

# Header with gradient animated title
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="gradient-text" style="font-size: 3.5rem; font-weight: bold; margin: 0;">
        📚 AI-Powered Reading Assistant
    </h1>
    <p style="font-size: 1.5rem; color: rgba(255, 255, 255, 0.9); margin-top: 1rem;">
        Empowering Dyslexic Students with Technology ✨
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for user info and settings
with st.sidebar:
    st.header("👤 User Settings")
    username = st.text_input("Enter your name:", value="Student")
    
    st.header("🔧 TTS Settings")
    slow_letters = st.checkbox("Slow letter pronunciation", value=True)
    slow_word = st.checkbox("Slow word pronunciation", value=False)
    
    st.header("📊 Progress")
    st.info("Keep practicing to improve your reading skills!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📷 Camera Interface")
    
    # Camera controls
    if st.button("🎥 Start Camera", key="start_camera"):
        st.session_state.camera_active = True
    
    if st.button("⏹️ Stop Camera", key="stop_camera"):
        st.session_state.camera_active = False
    
    # Camera display
    if st.session_state.get('camera_active', False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam")
        else:
            st.write("✅ Camera is active")
            
            camera_placeholder = st.empty()
            
            st.write("📺 **Live Camera Feed:**")
            ret, frame = cap.read()
            if ret:
                camera_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")
            
            col_cam1, col_cam2, col_cam3 = st.columns(3)
            
            with col_cam1:
                if st.button("📸 Capture Frame", key="capture_frame"):
                    ret, frame = cap.read()
                    if ret:
                        st.session_state.last_frame = frame
                        st.success("📸 Frame captured!")
                    else:
                        st.error("❌ Failed to capture frame")
            
            with col_cam2:
                if st.button("🔄 Recapture Frame", key="recapture_frame"):
                    ret, frame = cap.read()
                    if ret:
                        st.session_state.last_frame = frame
                        st.success("🔄 Frame recaptured!")
                    else:
                        st.error("❌ Failed to recapture frame")
            
            with col_cam3:
                if st.button("🔍 Detect Objects", key="detect_objects"):
                    if 'last_frame' in st.session_state:
                        st.write("🔄 Processing captured image...")
                        with st.spinner("Analyzing image with Gemini AI..."):
                            detected = get_object_detection_gemini(st.session_state.last_frame)
                        if detected and detected.strip():
                            st.session_state.detected_text = detected.strip()
                            st.session_state.current_word = detected.strip()
                            st.success(f"🎯 Detected: **{detected.strip()}**")
                        else:
                            st.error("⚠️ No objects detected. Try capturing a clearer frame.")
                            st.info("💡 **Troubleshooting tips:**")
                            st.write("- Make sure your internet connection is stable")
                            st.write("- Try holding the object closer to the camera")
                            st.write("- Ensure good lighting")
                            st.write("- Try a different object")
                    else:
                        st.error("❌ Please capture a frame first!")
            
            # Show captured frame if available
            if 'last_frame' in st.session_state:
                st.image(st.session_state.last_frame, channels="BGR", caption="Captured Frame")
                
                # Display detected object name in large dyslexic font
                if st.session_state.get('current_word'):
                    st.markdown("---")
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <h1 style="font-family: 'Comic Sans MS', 'Comfortaa', cursive; 
                                   font-size: 48px; 
                                   font-weight: bold; 
                                   color: #2E86AB; 
                                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                                   letter-spacing: 3px;">
                            {st.session_state.current_word.upper()}
                        </h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio Features Section
                    st.markdown("---")
                    st.header("🎵 Audio Features")
                    
                    # TTS Controls
                    st.subheader("🔊 Text-to-Speech")
                    st.write(f"Current word: **{st.session_state.current_word}**")
                    
                    col_tts1, col_tts2 = st.columns(2)
                    
                    with col_tts1:
                        if st.button("🔤 Spell and Read Word", key="spell_word"):
                            if st.session_state.current_word:
                                try:
                                    spell_word_with_highlighting(
                                        st.session_state.current_word,
                                        slow_letters=slow_letters,
                                        slow_word=slow_word
                                    )
                                    st.success("🔤 Word spelled and read!")
                                except Exception as e:
                                    st.error(f"❌ TTS Error: {e}")
                            else:
                                st.error("❌ No word to spell.")
                    
                    with col_tts2:
                        if st.button("🔢 Read Word Only", key="read_word_only"):
                            if st.session_state.current_word:
                                with st.spinner("🔊 Speaking..."):
                                    try:
                                        tts = gTTS(
                                            text=st.session_state.current_word,
                                            lang='en',
                                            slow=slow_word
                                        )
                                        
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                                            tts.save(f.name)
                                            playsound(f.name)
                                            os.remove(f.name)
                                        
                                        st.success("🔢 Word read!")
                                    except Exception as e:
                                        st.error(f"❌ TTS Error: {e}")
                            else:
                                st.error("❌ No word to read.")
                    
                    # Pronunciation feedback
                    if st.button("💡 Get Pronunciation Help", key="pronunciation_help"):
                        with st.spinner("Getting pronunciation help..."):
                            feedback = get_pronunciation_feedback(st.session_state.current_word)
                        
                        st.subheader("💡 Pronunciation Help")
                        st.markdown(feedback)
                    
                    # ========================================
                    # 🎤 ADVANCED SPEECH RECOGNITION SECTION
                    # ========================================
                    st.markdown("---")
                    st.header("🎤 Speech Recognition Practice")
                    
                    st.write(f"Practice pronouncing: **{st.session_state.current_word}**")
                    
                    # Pre-populate word in text field
                    st.markdown(f"""
                    <div style='text-align:center; margin:20px 0; padding:20px;
                                background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
                                border-radius:15px; border:2px solid rgba(255,255,255,0.2);'>
                        <p style='font-size:20px; color:#fff; margin-bottom:10px; font-family: "Comic Sans MS", "Comfortaa", cursive;'>
                            📝 Word to practice:
                        </p>
                        <p style='font-size:48px; font-weight:bold; color:#2E86AB; letter-spacing:8px;
                                  text-shadow:2px 2px 6px rgba(0,0,0,0.3); font-family: "Comic Sans MS", "Comfortaa", cursive;'>
                            {st.session_state.current_word.upper()}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Instructions
                    st.info("🎯 **How it works:** Click the button below, then speak the word letter by letter. Watch as each letter lights up when you say it correctly!")
                    
                    # Speech recognition button with advanced highlighting
                    if st.button("🎤 Start Advanced Speech Recognition", key="start_advanced_speech_recognition"):
                        st.markdown("---")
                        feedback = recognize_speech_unified(st.session_state.current_word, mode="advanced")
                            
                        if feedback:
                            st.success("✅ Speech recognition completed!")
                            
                            # Display final feedback summary
                            st.markdown("---")
                            st.subheader("📊 Final Results:")
                            
                            # Create a formatted display of the word with correct letters highlighted
                            word_display = ""
                            for i, (letter, status) in enumerate(feedback.items()):
                                if status == "correct":
                                    word_display += f"<span class='letter-correct'>{letter}</span>"
                                else:
                                    word_display += f"<span class='letter-incorrect'>{letter}</span>"
                                word_display += " "
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin: 20px 0;">
                                <h3 style="font-family: 'Comic Sans MS', 'Comfortaa', cursive;">
                                    Your pronunciation: {word_display}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Individual letter feedback in columns
                            feedback_cols = st.columns(len(feedback))
                            for i, (letter, status) in enumerate(feedback.items()):
                                with feedback_cols[i]:
                                    if status == "correct":
                                        st.success(f"✅ {letter}")
                                    else:
                                        st.error(f"❌ {letter}")
                            
                            # Calculate accuracy
                            correct_count = sum(1 for status in feedback.values() if status == "correct")
                            accuracy = (correct_count / len(feedback)) * 100
                            
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                            
                            # Provide encouragement
                            if accuracy >= 80:
                                st.success("🌟 Excellent work! You're doing great!")
                            elif accuracy >= 60:
                                st.info("👍 Good job! Keep practicing!")
                            else:
                                st.info("💪 Keep trying! Practice makes perfect!")
                        else:
                            st.error("❌ Speech recognition failed. Please try again.")
                    
                    # Alternative: Basic speech recognition (fallback)
                    st.markdown("---")
                    st.subheader("🎙️ Basic Speech Recognition (Fallback)")
                    if st.button("🎙️ Use Basic Recognition", key="basic_speech_recognition"):
                        with st.spinner("🎤 Listening... Please speak now!"):
                            feedback = recognize_speech(st.session_state.current_word)
                            
                        if feedback:
                            st.success("✅ Speech recognition completed!")
                            
                            # Display letter-by-letter feedback
                            st.subheader("📊 Letter-by-Letter Feedback:")
                            
                            word_display = ""
                            for i, (letter, status) in enumerate(feedback.items()):
                                if status == "correct":
                                    word_display += f"<span class='letter-correct'>{letter}</span>"
                                else:
                                    word_display += f"<span class='letter-incorrect'>{letter}</span>"
                                word_display += " "
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin: 20px 0;">
                                <h3 style="font-family: 'Comic Sans MS', 'Comfortaa', cursive;">
                                    Your pronunciation: {word_display}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            feedback_cols = st.columns(len(feedback))
                            for i, (letter, status) in enumerate(feedback.items()):
                                with feedback_cols[i]:
                                    if status == "correct":
                                        st.success(f"✅ {letter}")
                                    else:
                                        st.error(f"❌ {letter}")
                            
                            correct_count = sum(1 for status in feedback.values() if status == "correct")
                            accuracy = (correct_count / len(feedback)) * 100
                            
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                            
                            if accuracy >= 80:
                                st.success("🌟 Excellent work! You're doing great!")
                            elif accuracy >= 60:
                                st.info("👍 Good job! Keep practicing!")
                            else:
                                st.info("💪 Keep trying! Practice makes perfect!")
                        else:
                            st.error("❌ Speech recognition failed. Please try again.")