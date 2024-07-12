import cv2
import numpy as np
import imghdr
import smtplib
from email.message import EmailMessage
from flask import Flask, render_template, Response
import time
import logging
from werkzeug.serving import run_simple

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Initialize the email parameters
FROM_EMAIL = "oden.raspi@gmail.com"

# Read password from file
with open('notapasswordfileIswear.txt', 'r') as f:
    FROM_PASSWORD = f.readline().strip()

TO_EMAIL = "kheredia97@gmail.com"
SUBJECT = "Dog detected"
BODY = "A dog has been detected. Please see the attached picture. Woof woof!"

# Initialize the web server
app = Flask(__name__)

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, int(60))
    return cap

@app.route('/')
def index():
    return render_template('index.html')

def video_streaming():
    cap = initialize_camera()
    last_sent_time = time.time() - 1800  # 1800 seconds = 30 minutes

    while True:
        try:
            ret, frame = cap.read()
            
            # Prepare the frame for object detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            
            # Pass the blob through the network and obtain the detections
            net.setInput(blob)
            detections = net.forward()
            
            dog_detected = False
            
            # Loop over the detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections
                if confidence > 0.5:
                    class_id = int(detections[0, 0, i, 1])
                    
                    # Class 12 represents the 'dog' class in MobileNet-SSD
                    if class_id == 12:
                        dog_detected = True
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, 'Dog', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if dog_detected and (time.time() - last_sent_time > 1800):
                last_sent_time = time.time()
                
                # Take a picture
                img_name = "dog_detected.jpg"
                cv2.imwrite(img_name, frame)
                
                # Send an email with the picture as an attachment
                msg = EmailMessage()
                msg['Subject'] = SUBJECT
                msg['From'] = FROM_EMAIL
                msg['To'] = TO_EMAIL
                msg.set_content(BODY)
                
                # Add the picture as an attachment
                with open(img_name, 'rb') as f:
                    file_data = f.read()
                    file_type = imghdr.what(f.name)
                    file_name = f.name
                msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)
                
                # Send the email via SMTP
                try:
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(FROM_EMAIL, FROM_PASSWORD)
                        print('Connection Established.')
                        smtp.send_message(msg)
                        print('Email Sent.')
                except Exception as e:
                    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

            # Encode the frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            logging.error(f"Error in video_streaming: {e}")
            cap.release()
            cap = initialize_camera()

@app.route('/video_feed')
def video_feed():
    return Response(video_streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_app():
    while True:
        try:
            with open('forsurenotmyIP.txt', 'r') as f:
                ip = f.readline().strip()
            run_simple(ip, 8000, app, use_reloader=False, use_debugger=False)
        except Exception as e:
            logging.error(f"Server error: {e}")
            print(f"Server error occurred. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == '__main__':
    run_app()