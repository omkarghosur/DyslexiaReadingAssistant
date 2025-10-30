import cv2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# ================================
#  STEP 1: Gemini Object Detection
# ================================

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open webcam")

print("✅ Webcam started successfully.")
print("👉 Press SPACE to detect objects, or Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Show live video feed
    cv2.imshow("Gemini Object Detection (Press SPACE to Detect)", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # When SPACE is pressed → Capture and detect objects
    if key == ord(' '):
        print("\n📸 Capturing frame and sending to Gemini...")
        try:
            # Convert OpenCV frame (BGR) → RGB → PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Detection prompt
            prompt = (
                "Identify only the main object being shown by the person in the image. "
                "Ignore human body parts (like hand, face, eyes, T-shirt, etc.) and background items. "
                "Describe only the primary object that seems to be intentionally displayed or held up."
                "Do NOT mention colors, background, people, hands, clothes, or any body parts. "
                "Return only the object names like 'Apple', 'Book', 'Eyeglasses', etc., as bullet points."
            )

            # Send image + prompt to Gemini Vision model
            response = model.generate_content([prompt, img])

            # Display results
            if response and response.text:
                detected = response.text.strip()
                print("🧠 Detected objects:\n", detected)
            else:
                print("⚠️ No objects detected or empty response.")
        except Exception as e:
            print("⚠️ Error calling Gemini API:", e)

    # Press Q to quit
    elif key == ord('q'):
        print("👋 Exiting program...")
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
