# 🧠 Real-Time Sign Language Detection using YOLOv8 + Flask
This project is a real-time sign language detection web application built using Flask, YOLOv8, and OpenCV. It uses a webcam to capture hand gestures and identifies the corresponding English alphabet in real-time using a trained YOLO model.

---

## 📁 Project Structure
.
├── static/                  # Contains all the sign language alphabet images
│   ├── A.png to Z.png       # Reference images for each letter
│   └── signlanguageimage... # Additional images for the Learn page
├── templates/               # HTML templates (index.html, signlanguage.html, learn.html)
├── app.py                   # Main Flask application
├── best (3).pt              # Trained YOLOv8 model weights
├── a.py                     # (optional) extra script
├── venv/                    # Python virtual environment

---

## 🚀 Features
📷 Real-time webcam-based letter detection

🤖 YOLOv8-powered gesture recognition

🌐 Flask-based web interface

📚 Learn tab with hand sign references

🔁 Live video stream and AJAX updates

---

## 📄 Pages Overview
/ - Home page

/signlanguage - Real-time detection view

/learn - View all hand sign images

/video_feed - Live video stream

/get_letter - AJAX endpoint for current predicted letter

---

## 📦 Model Training (Optional)
If you want to train your own model:

Collect sign language hand gesture images for each letter.

Annotate them using tools like Roboflow or LabelImg.

Train with Ultralytics YOLOv8 CLI or Python API.

Save the model as best.pt and replace best (3).pt.

