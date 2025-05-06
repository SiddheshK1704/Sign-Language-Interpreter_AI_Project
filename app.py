from flask import Flask, render_template, Response, jsonify, url_for, request
import cv2
from ultralytics import YOLO
import threading
import time
import os

# Create Flask app with proper template and static folder configuration
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Global variables for sharing data between threads
current_frame = None
current_letter = ""
detection_active = False
lock = threading.Lock()

# Load YOLO model
model = YOLO(r'best (3).pt')

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise Exception("Could not open video device")
        
        # Set resolution
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global current_frame, current_letter
        
        success, frame = self.video.read()
        if not success:
            return None
            
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
            
        # Make predictions with YOLO
        results = model.predict(source=frame, conf=0.3)
        
        detected_letter = ""
        for res in results:
            for box in res.boxes:
                # Extract coordinates and other values
                x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                
                class_id = int(box.cls[0].cpu().numpy())
                class_name = res.names.get(class_id, "Unknown")
                
                detected_letter = class_name
                
                xmin = x_center - (width / 2)
                ymin = y_center - (height / 2)
                xmax = x_center + (width / 2)
                ymax = y_center + (height / 2)
                
                confidence = round(box.conf[0].item(), 3)
                frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        
        # Add text to the frame if a letter is detected
        if detected_letter:
            cv2.putText(frame, detected_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # Update current letter
            with lock:
                current_letter = detected_letter
        
        # Encode the frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        
        with lock:
            current_frame = frame_bytes
            
        return frame_bytes

# Background thread to continuously process frames
def background_processing():
    global detection_active, current_frame, current_letter
    camera = None
    
    while True:
        with lock:
            active = detection_active
        
        if active:
            if camera is None:
                try:
                    camera = VideoCamera()
                except Exception as e:
                    print(f"Error initializing camera: {e}")
                    with lock:
                        detection_active = False
                    time.sleep(5)
                    continue
                    
            try:
                camera.get_frame()
            except Exception as e:
                print(f"Error getting frame: {e}")
                camera = None
        else:
            if camera is not None:
                del camera
                camera = None
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signlanguage')
def signlanguage():
    global detection_active
    with lock:
        detection_active = True
    return render_template('signlanguage.html')

@app.route('/activate_detection', methods=['POST'])
def activate_detection():
    global detection_active
    with lock:
        detection_active = True
    return jsonify({'status': 'success'})

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/get_letter')
def get_letter():
    global current_letter
    with lock:
        letter = current_letter
    return jsonify({'letter': letter})

def gen():
    global current_frame
    while True:
        with lock:
            frame = current_frame
            active = detection_active
        
        if frame is not None and active:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # If no frame is available, send a blank frame
            blank_frame = generate_blank_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

def generate_blank_frame():
    # Create a blank frame
    blank = cv2.imread('static/images/waiting.jpg') if os.path.exists('static/images/waiting.jpg') else None
    
    if blank is None:
        # Create a blank image with message
        blank = cv2.cvtColor(cv2.imread_from_memory(bytearray(width=640, height=480, channels=3)),
                            cv2.COLOR_BGR2RGB)
        cv2.putText(blank, "Camera starting...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
    
    # Encode the blank frame to JPEG
    _, jpeg = cv2.imencode('.jpg', blank)
    return jpeg.tobytes()

@app.route('/video_feed')
def video_feed():
    global detection_active
    # Ensure detection is active when video feed is requested
    with lock:
        detection_active = True
    return Response(gen(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Ensure directory structure exists
def create_directory_structure():
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Create static/images directory if it doesn't exist
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

if __name__ == '__main__':
    # Create directory structure
    create_directory_structure()
    
    # Start background thread for processing video
    processing_thread = threading.Thread(target=background_processing)
    processing_thread.daemon = True
    processing_thread.start()
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)