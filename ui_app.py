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
import difflib

# Import our custom modules
from tts_module import DyslexiaTTS
from speech_module import recognize_speech_unified

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
    """
    Spell word letter-by-letter with synchronized visual highlighting and audio.
    Updated: Inactive letters remain simple white/normal. Active letter becomes Bold/Red.
    Spaces are now SILENT (no audio "space").
    """
    placeholder = st.empty()
    word_upper = word.upper()
    
    # Step 1: Spell each letter with synchronized audio + visual
    for i, letter in enumerate(word_upper):
        
        html_parts = []
        for j, char in enumerate(word_upper):
            if j == i:
                # ACTIVE LETTER: Bold, Red, Slightly Larger
                html_parts.append(
                    f"<span style='color:#FF4B4B; font-size:80px; font-weight:bold; "
                    f"text-shadow:2px 2px 4px rgba(0,0,0,0.3); font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{char}</span>"
                )
            else:
                # INACTIVE LETTER: White, Normal weight (Not Bold), Stable
                html_parts.append(
                    f"<span style='color:#FFFFFF; font-size:60px; font-weight:normal; "
                    f"font-family: \"Comic Sans MS\", \"Comfortaa\", cursive;'>{char}</span>"
                )
        
        # Display the word
        full_html = f"""
        <div style='text-align:center; letter-spacing:15px; margin:30px 0;'>
            {''.join(html_parts)}
        </div>
        """
        placeholder.markdown(full_html, unsafe_allow_html=True)
        
        # Play audio (SILENT FOR SPACES)
        try:
            if letter == ' ':
                # Do not play audio, just pause
                time.sleep(0.3)
            else:
                letter_tts = gTTS(text=letter, lang='en', slow=slow_letters)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    letter_tts.save(f.name)
                    playsound(f.name)
                    os.remove(f.name)
        except Exception as e:
            st.error(f"Audio error: {e}")
        
        time.sleep(0.2)
    
    # Step 2: Show full word (Standard Blue) and speak it
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
    
    try:
        word_tts = gTTS(text=word, lang='en', slow=slow_word)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            word_tts.save(f.name)
            playsound(f.name)
            os.remove(f.name)
    except Exception as e:
        st.error(f"Audio error: {e}")

# ========================================
# OBJECT DETECTION FUNCTIONS
# ========================================
def get_object_detection_gemini(frame):
    """Detects objects in image using Gemini API."""
    if not api_key: return None
    
    try:
        # Resize if huge
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame = cv2.resize(frame, (800, int(height * scale)))
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Look at this image carefully. Identify the main object that a person is showing or holding. Return only the single most prominent object name in English (e.g. 'Apple', 'Book'). Be specific and concise."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
                ]
            }]
        }
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            candidates = response.json().get('candidates', [])
            if candidates and 'content' in candidates[0]:
                text = candidates[0]['content']['parts'][0]['text']
                
                # Cleanup text
                detected_text = text.strip()
                prefixes = ["The object is", "I can see", "This is", "Object:"]
                for p in prefixes:
                    if detected_text.lower().startswith(p.lower()):
                        detected_text = detected_text[len(p):].strip()
                return detected_text.split('.')[0].strip()
            
    except Exception as e:
        st.error(f"Detection Error: {e}")
    return None

def get_pronunciation_feedback(word):
    """Gets letter-by-letter pronunciation feedback from Gemini."""
    if not word or not api_key: return "Feedback unavailable."
    
    try:
        payload = {
            "contents": [{
                "parts": [{"text": f"Explain how to pronounce '{word}' letter by letter for a dyslexic student. Simple English. No complex phonetics."}]
            }]
        }
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return "Could not get feedback."

# ========================================
# STREAMLIT UI
# ========================================

st.set_page_config(page_title="AI Reading Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* { font-family: "Comic Sans MS", "Comfortaa", cursive !important; }
.stButton>button { background: linear-gradient(135deg,#667eea,#764ba2); color:#fff !important; border-radius:12px; font-weight:bold; }
h1,h2 { background: linear-gradient(135deg,#667eea 0%, #f093fb 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>
""", unsafe_allow_html=True)

# Session State Init
if 'session_id' not in st.session_state: st.session_state.session_id = str(int(time.time()))
if 'current_word' not in st.session_state: st.session_state.current_word = ""

# Header
st.markdown("<h1 style='text-align: center;'>📚 AI-Powered Reading Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("👤 Settings")
    username = st.text_input("Name:", value="Student")
    slow_letters = st.checkbox("Slow Letters", value=True)
    slow_word = st.checkbox("Slow Word", value=False)

# ==========================================
# 🟢 COLUMNS
# ==========================================
col1, col2 = st.columns([1.5, 1])

# --- COLUMN 1: CAMERA ---
with col1:
    st.header("📷 Camera Interface")
    
    img_file_buffer = st.camera_input("📸 Take a Photo")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.session_state.last_frame = cv2_img
        
        if st.button("🔍 Detect Objects", key="detect_objects"):
            st.write("🔄 Processing...")
            with st.spinner("Analyzing image with Gemini AI..."):
                detected = get_object_detection_gemini(st.session_state.last_frame)
            
            if detected:
                st.session_state.current_word = detected
                st.session_state.detected_text = detected
                st.success(f"🎯 Detected: **{detected}**")
                st.rerun() 
            else:
                st.error("⚠️ No objects detected.")

# --- COLUMN 2: MANUAL INPUT ---
with col2:
    st.header("📝 Manual Input")
    
    manual_word = st.text_input("Type a word:", key="manual_input_box")
    if st.button("🎯 Practice Typed Word"):
        if manual_word.strip():
            st.session_state.current_word = manual_word.strip()
            st.rerun()
    
    st.markdown("---")
    st.subheader("Or choose one:")
    quick_words = ["CAT", "DOG", "PHONE", "WATER", "COMPUTER"]
    for word in quick_words:
        if st.button(word):
            st.session_state.current_word = word
            st.rerun()
    
    st.markdown("---")
    if st.session_state.current_word:
        st.info(f"**Current Word:** {st.session_state.current_word}")

# ==========================================
# 🟢 RESULTS & PRACTICE SECTION (FULL WIDTH)
# ==========================================
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
    
    # Pronunciation feedback
    if st.button("💡 Get Pronunciation Help", key="pronunciation_help"):
        with st.spinner("Getting pronunciation help..."):
            feedback = get_pronunciation_feedback(st.session_state.current_word)
        
        st.subheader("💡 Pronunciation Help")
        st.markdown(feedback)
    
    # ========================================
    # 🎤 SPEECH RECOGNITION SECTION
    # ========================================
    st.markdown("---")
    st.header("🎤 Speech Recognition Practice")
    
    # Word Display Card
    st.markdown(f"""
    <div style='text-align:center; margin:10px 0; padding:15px;
                background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
                border-radius:15px; border:1px solid rgba(255,255,255,0.2);'>
        <p style='font-size:18px; color:#ddd; margin-bottom:5px;'>Target Word:</p>
        <p style='font-size:40px; font-weight:bold; color:#fff; letter-spacing:5px; margin:0;'>
            {st.session_state.current_word.upper()}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two columns for the two different modes
    col_speech1, col_speech2 = st.columns(2)
    
    # --- MODE 1: SPELLING (Letter by Letter) ---
    with col_speech1:
        st.subheader("🔤 Spelling")
        st.caption("Say: C... A... T...")
        if st.button("🎤 Practice Spelling", key="btn_spell_practice"):
            feedback = recognize_speech_unified(
                st.session_state.current_word, 
                mode="spelling", 
                slow_speed=slow_word
            )
            if feedback:
                # Calculate Score
                correct = sum(1 for s in feedback.values() if s == "correct")
                total = len(feedback)
                st.metric("Spelling Accuracy", f"{int(correct/total*100)}%")

    # --- MODE 2: PRONUNCIATION (Whole Word) ---
    with col_speech2:
        st.subheader("🗣️ Pronunciation")
        st.caption(f"Say: {st.session_state.current_word}")
        if st.button("🎤 Practice Speaking", key="btn_pronounce_practice"):
            feedback = recognize_speech_unified(
                st.session_state.current_word, 
                mode="pronunciation", 
                slow_speed=slow_word
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: rgba(255, 255, 255, 0.7);">
    <p style="font-size: 1.1rem; font-family: 'Comic Sans MS', 'Comfortaa', cursive;">
        💙 Made with love for dyslexic learners 💙
    </p>
    <p style="font-size: 0.9rem; font-family: 'Comic Sans MS', 'Comfortaa', cursive;">
        Keep practicing, you're doing amazing! ✨
    </p>
</div>
""", unsafe_allow_html=True)