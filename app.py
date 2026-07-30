from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

# Load known faces and their encodings
known_faces = {}
known_faces_names = []

# Function to load images and encode faces
def load_and_encode_image(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if len(encoding) > 0:
        known_faces[name] = encoding[0]
        known_faces_names.append(name)
    else:
        print("No faces found in", name)

# Define image paths and names
image_paths_and_names = [
    ("static/photos/tata.jpeg", "Ratan TATA"),
    ("static/photos/sadmona.jpg", "Monolisa"),
    ("static/photos/tesla.jpg", "Nicolo Tesla"),
    ("static/photos/jobs.jpg", "Steve Jobs")
]

# Load and encode images
for image_path, name in image_paths_and_names:
    load_and_encode_image(image_path, name)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Create or open CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = open(current_date + '.csv', 'w+', newline='')
csv_writer = csv.writer(csv_file)

# Set to store names of students already marked present in the current frame
students_present_in_frame = set()

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert frame to RGB
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Initialize list to store recognized face names
        face_names = []

        # Loop through each face encoding in the current frame
        for face_encoding in face_encodings:
            # Check if the current face matches any of the known faces
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"

            # If a match is found, retrieve the name of the known face
            if True in matches:
                match_index = matches.index(True)
                name = known_faces_names[match_index]

            # Append recognized face name to list
            face_names.append(name)

            # Write attendance record to CSV if the recognized face is in the list of known faces
            if name in known_faces_names and name not in students_present_in_frame:
                current_time = now.strftime("%H:%M:%S")
                csv_writer.writerow([name, current_time])
                students_present_in_frame.add(name)

        # Display recognized faces and names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,128,0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
