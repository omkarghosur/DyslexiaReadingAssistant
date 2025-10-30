import streamlit as st
import cv2
import google.generativeai as genai
import os
import base64
import requests
import threading
import time
import pyttsx3
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import our custom modules
from tts_module import DyslexiaTTS
from speech_module import recognize_speech
from progress_db import ProgressDB

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

# Initialize progress database - Temporarily disabled
# progress_db = ProgressDB()

# ---- TTS Utilities (female voice selection) ----
def init_tts_engine(rate: int = 160, volume: float = 1.0):
    """Create and configure a pyttsx3 engine with a female voice when available.

    Works on macOS (NSSpeechSynthesizer), Windows (SAPI5), Linux (espeak).
    Falls back to the default voice if no explicit female voice is found.
    """
    try:
        engine = pyttsx3.init()
        # Rate and volume
        try:
            engine.setProperty('rate', int(rate))
        except Exception:
            pass
        try:
            engine.setProperty('volume', float(volume))
        except Exception:
            pass

        # Attempt to pick a female voice
        try:
            voices = engine.getProperty('voices') or []
            selected = None
            preferred_keywords = [
                'female', 'samantha', 'victoria', 'tessa', 'karen', 'serena',
                'zira', 'eva', 'susan', 'heather'
            ]
            for v in voices:
                vname = (getattr(v, 'name', '') or '').lower()
                vid = (getattr(v, 'id', '') or '').lower()
                if any(k in vname or k in vid for k in preferred_keywords):
                    selected = v
                    break
            if selected is None and voices:
                # fallback to the first voice to ensure speaking works
                selected = voices[0]
            if selected is not None:
                engine.setProperty('voice', selected.id)
        except Exception:
            pass
        return engine
    except Exception:
        return None


def ensure_tts_engine(rate: int, volume: float):
    """Singleton TTS engine stored in Streamlit session_state."""
    key = "_tts_engine"
    if key not in st.session_state or st.session_state[key] is None:
        st.session_state[key] = init_tts_engine(rate=rate, volume=volume)
        # Remember current settings
        st.session_state["_tts_rate"] = rate
        st.session_state["_tts_vol"] = volume
    else:
        # Update properties if changed
        try:
            if st.session_state.get("_tts_rate") != rate:
                st.session_state[key].setProperty('rate', int(rate))
                st.session_state["_tts_rate"] = rate
            if st.session_state.get("_tts_vol") != volume:
                st.session_state[key].setProperty('volume', float(volume))
                st.session_state["_tts_vol"] = volume
        except Exception:
            pass
    return st.session_state[key]


def speak_text_or_fallback(text: str, rate: int, volume: float):
    """Try pyttsx3; if it fails (common on some macOS envs), fallback to 'say'."""
    engine = ensure_tts_engine(rate=rate, volume=volume)
    if engine is not None:
        try:
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            pass
    # Fallback for macOS using built-in 'say'
    try:
        import platform, subprocess, shlex
        if platform.system() == 'Darwin':
            # Prefer Samantha if available
            cmd = f"say -v Samantha {shlex.quote(text)}"
            subprocess.run(cmd, shell=True, check=True)
            return True
    except Exception:
        pass
    return False

# Object Detection Functions
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
    import time
    
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
            
            # Create session with custom SSL settings
            session = requests.Session()
            session.verify = True  # Enable SSL verification
            
            response = session.post(api_url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            
            candidates = result.get('candidates', [])
            if candidates and 'content' in candidates[0]:
                parts = candidates[0]['content'].get('parts', [])
                text = parts[0].get('text', '') if parts else ''
                detected_text = text.strip()
                
                # Clean up the response - remove common prefixes
                prefixes_to_remove = ["The object is", "I can see", "This is", "The main object is", "Object:"]
                for prefix in prefixes_to_remove:
                    if detected_text.lower().startswith(prefix.lower()):
                        detected_text = detected_text[len(prefix):].strip()
                
                # Remove punctuation and extra words
                detected_text = detected_text.split('.')[0].split(',')[0].strip()
                
                return detected_text if detected_text else None
            
            return None
            
        except requests.exceptions.SSLError as e:
            st.warning(f"🔄 SSL Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                st.error("❌ SSL connection failed after multiple attempts. Please check your internet connection.")
                return None
                
        except requests.exceptions.RequestException as e:
            st.warning(f"🔄 Request Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                st.error(f"❌ API request failed after multiple attempts: {e}")
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
                        "text": f"For a student with dyslexia reading '{word}': "
                                "Provide interactive pronunciation feedback. "
                                "Break it down letter by letter with phonetic sounds (e.g., 'c-a-t: /k/ /æ/ /t/'). "
                                "Keep it short, encouraging, and dyslexia-friendly."
                    }
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
            return parts[0].get('text', 'No feedback received.') if parts else 'No feedback received.'
        return "No feedback received."
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

# Add custom CSS (Fixed Streamlit Interface - simplified and stable)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');

/* Simplified animated background */
@keyframes gradient { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }
#root { background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe); background-size: 400% 400%; animation: gradient 15s ease infinite; min-height: 100vh; }

/* Glassmorphism card */
.main-container, .glass-card { background: rgba(255,255,255,0.15); backdrop-filter: blur(16px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2); padding: 1.5rem; }

/* Buttons (reduced animation) */
.stButton>button { background: linear-gradient(135deg,#667eea,#764ba2); border:none; border-radius:12px; color:#fff !important; font-weight:600; padding:0.5rem 1.25rem; box-shadow:0 4px 15px rgba(116,79,168,0.45); }
.stButton>button:hover { transform: translateY(-2px); background: linear-gradient(135deg,#764ba2,#f093fb); }

/* Typography (streamlined) */
h1,h2,h3,h4,h5,h6 { font-family: 'Comfortaa', cursive !important; background: linear-gradient(135deg,#667eea 0%, #f093fb 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:2px; }

/* Letter feedback */
.letter-correct { background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%); color:#000; padding:6px 10px; border-radius:10px; font-weight:700; display:inline-block; }
.letter-incorrect { background: linear-gradient(135deg,#fa709a 0%,#fee140 100%); color:#000; padding:6px 10px; border-radius:10px; font-weight:700; display:inline-block; }

/* Inputs */
.stTextInput>div>div>input { background: rgba(255,255,255,0.2); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.3); border-radius:10px; color:#fff; }

/* Scrollbar */
::-webkit-scrollbar { width:10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# Removed complex JS interactions to keep interface stable

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_" + str(int(time.time()))
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + str(int(time.time()))
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""

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
    # progress_db.update_user(st.session_state.user_id, username)
    
    st.header("🔧 TTS Settings")
    rate_letters = st.slider("Letter Reading Speed", 100, 200, 150)
    rate_word = st.slider("Word Reading Speed", 120, 200, 160)
    volume = st.slider("Volume", 0.1, 1.0, 1.0)
    
    st.header("📊 Progress")
    st.info("Progress tracking will be available after adding your API key")

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
            
            # Live camera feed - always on when camera is active
            camera_placeholder = st.empty()
            
            # Show live camera feed continuously
            st.write("📺 **Live Camera Feed:**")
            ret, frame = cap.read()
            if ret:
                camera_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")
            
            # Camera controls
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
                        <h1 style="font-family: 'OpenDyslexic', Arial, sans-serif; 
                                   font-size: 48px; 
                                   font-weight: bold; 
                                   color: #2E86AB; 
                                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                                   letter-spacing: 3px;">
                            {st.session_state.current_word.upper()}
                        </h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio Features Section - moved below captured frame
                    st.markdown("---")
                    st.header("🎵 Audio Features")
                    
                    # TTS Controls
                    st.subheader("🔊 Text-to-Speech")
                    st.write(f"Current word: **{st.session_state.current_word}**")
                    
                    col_tts1, col_tts2 = st.columns(2)
                    
                    with col_tts1:
                        if st.button("🔤 Spell Letters", key="spell_letters"):
                            if st.session_state.current_word:
                                word = st.session_state.current_word.upper()

                                # Create a placeholder for visual progress
                                spelling_placeholder = st.empty()

                                # Initialize one engine with female voice
                                import time as _time
                                engine = ensure_tts_engine(rate=rate_letters, volume=volume)
                                if engine is None:
                                    st.error("TTS engine not available on this system.")
                                else:
                                    # Speak letter-by-letter with highlight (lowercase audio)
                                    for i, letter in enumerate(word):
                                        # Visual display with current letter highlighted
                                        word_display = ""
                                        for j, char in enumerate(word):
                                            disp_char = char.upper()
                                            if j == i:
                                                word_display += f"<span style='background-color:#FFD700;color:#000;padding:5px;border-radius:5px;font-weight:bold;font-size:24px;'>{disp_char}</span>"
                                            elif j < i:
                                                word_display += f"<span style='background-color:#90EE90;color:#000;padding:5px;border-radius:5px;font-weight:bold;font-size:20px;'>{disp_char}</span>"
                                            else:
                                                word_display += f"<span style='color:#666;font-size:18px;'>{disp_char}</span>"
                                            word_display += " "
                                        spelling_placeholder.markdown(
                                            f"<div style='text-align:center;margin:10px 0;'><h2 style=\"font-family: 'OpenDyslexic', Arial, sans-serif;\">{word_display}</h2></div>",
                                            unsafe_allow_html=True,
                                        )
                                        ok = speak_text_or_fallback(letter.lower(), rate=rate_letters, volume=volume)
                                        if not ok:
                                            st.error("Audio output failed for this letter.")
                                        _time.sleep(0.2)

                                    # Final green display
                                    final_display = " ".join(
                                        f"<span style='background-color:#90EE90;color:#000;padding:5px;border-radius:5px;font-weight:bold;font-size:20px;'>{c.upper()}</span>"
                                        for c in word
                                    )
                                    spelling_placeholder.markdown(
                                        f"<div style='text-align:center;margin:10px 0;'><h2 style=\"font-family: 'OpenDyslexic', Arial, sans-serif;\">{final_display}</h2><p style='color:green;font-weight:bold;'>✅ Spelling Complete!</p></div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.success("🔤 Letters spelled!")
                            else:
                                st.error("❌ No word to spell. Please detect an object or enter a word first.")
                    
                    with col_tts2:
                        if st.button("📢 Read Word", key="read_word"):
                            if st.session_state.current_word:
                                ok = speak_text_or_fallback(st.session_state.current_word, rate=rate_word, volume=volume)
                                if ok:
                                    st.success("📢 Word read!")
                                else:
                                    st.error("Failed to speak. Check your audio output device.")
                            else:
                                st.error("❌ No word to read. Please detect an object or enter a word first.")
                    
                    # Pronunciation feedback
                    if st.button("💡 Get Pronunciation Help", key="pronunciation_help"):
                        feedback = get_pronunciation_feedback(st.session_state.current_word)
                        st.info(f"💡 **Pronunciation Help:**\n{feedback}")
                    
                    # Speech Recognition Section
                    st.markdown("---")
                    st.header("🎤 Speech Recognition Practice")
                    
                    st.write(f"Practice pronouncing: **{st.session_state.current_word}**")
                    
                    if st.button("🎤 Start Speech Recognition", key="start_speech_recognition"):
                        with st.spinner("🎤 Listening... Please speak now!"):
                            feedback = recognize_speech(st.session_state.current_word)
                            
                        if feedback:
                            st.success("✅ Speech recognition completed!")
                            
                            # Display letter-by-letter feedback with visual highlighting
                            st.subheader("📊 Letter-by-Letter Feedback:")
                            
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
                                <h3 style="font-family: 'OpenDyslexic', Arial, sans-serif;">
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
            
            # Manual input fallback
            st.markdown("---")
            st.subheader("📝 Manual Input (Fallback)")
            st.write("If object detection isn't working, you can manually enter a word to practice:")
            
            manual_word = st.text_input("Enter a word to practice:", key="manual_word_input")
            if st.button("✅ Use Manual Word", key="use_manual_word"):
                if manual_word and manual_word.strip():
                    st.session_state.detected_text = manual_word.strip()
                    st.session_state.current_word = manual_word.strip()
                    st.success(f"📝 Using manual word: **{manual_word.strip()}**")
                    st.write(f"🎯 Ready for pronunciation practice: {manual_word.strip()}")
                else:
                    st.error("Please enter a word first!")
            
            # Debug information
            if st.checkbox("🔧 Show Debug Info", key="debug_info"):
                st.write("**Debug Information:**")
                st.write(f"- Camera active: {st.session_state.get('camera_active', False)}")
                st.write(f"- Current word: {st.session_state.get('current_word', 'None')}")
                st.write(f"- Detected text: {st.session_state.get('detected_text', 'None')}")
                if 'last_frame' in st.session_state:
                    st.write(f"- Last frame shape: {st.session_state.last_frame.shape}")
        
        cap.release()

with col2:
    st.header("📊 Settings & Info")
    
    # TTS Settings
    st.subheader("🔧 TTS Settings")
    st.write("Adjust these settings to customize the speech:")
    st.write(f"- Letter Speed: {rate_letters}")
    st.write(f"- Word Speed: {rate_word}")
    st.write(f"- Volume: {volume}")
    
    # Instructions
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **🎥 Start Camera** - Opens your webcam
    2. **📸 Capture Frame** - Take a picture of your object
    3. **🔍 Detect Objects** - AI identifies the object
    4. **🔤 Spell Letters** - Hear each letter pronounced
    5. **🎤 Speech Recognition** - Practice pronunciation
    """)
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - Hold objects clearly in front of camera
    - Ensure good lighting for better detection
    - Speak clearly for speech recognition
    - Use manual input if detection fails
    """)

# Footer
st.markdown("---")
st.markdown("### 🎯 Features Available:")
st.markdown("""
- **📷 Object Detection**: Identify objects using your camera
- **📝 Text Extraction**: Extract text from images
- **🔊 Text-to-Speech**: Hear words spelled letter by letter
- **🎤 Speech Recognition**: Practice pronunciation with feedback
- **📊 Progress Tracking**: Monitor your learning progress
- **💡 AI Feedback**: Get pronunciation help from AI
""")