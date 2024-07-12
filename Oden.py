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

TO_EMAIL = "vincenzo.scotto001@gmail.com"
SUBJECT = "Human detected"
BODY = "A human has been detected. Please see the attached picture or video. Shall I smite it?"

# Initialize the web server
app = Flask(__name__)

# Load the HOG Descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, int(60))
    return cap

# Define the index page
@app.route('/')
def index():
    # Return the HTML page
    return render_template('index.html')

def video_streaming():
    cap = initialize_camera()
    last_sent_time = time.time() - 1800  # 1800 seconds = 30 minutes
    print(last_sent_time)

    while True:
        try:
            # Capture a frame from the camera
            ret, frame = cap.read()
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            hog_bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
            
            # If a body is detected, take a picture and send an email
            if (len(hog_bodies) > 0) and (time.time() - last_sent_time > 1800):
                # Update the time of the last sent email
                last_sent_time = time.time()
                
                # Take a picture
                img_name = "human_detected.jpg"
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
            
            # Draw rectangles around the detected bodies
            for (x,y,w,h) in hog_bodies:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

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