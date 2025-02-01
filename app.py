import cv2 as cv
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from keras.models import load_model

# Load pre-trained models
age_model = load_model("./Models/ageModel2.h5")
emotion_model = load_model("./Models/emotionModel3.h5")

# Load Haarcascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# CSV file to store detections
csv_file = "./CSV_File/detections.csv"

# Create CSV file if it doesn't exist
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Age", "Emotion"])
    df.to_csv(csv_file, index=False)

# Start webcam feed
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame[y:y + h, x:x + w]
        face_gray = gray[y:y + h, x:x + w]

        # Age prediction
        age_input = cv.resize(face_roi, (224, 224))  
        age_input = np.expand_dims(age_input, axis=0) / 255.0  
        _, predicted_age = age_model.predict(age_input)
        age = int(predicted_age[0][0])

        # Emotion prediction
        emotion_input = cv.resize(face_gray, (48, 48))  
        emotion_input = np.expand_dims(emotion_input, axis=[0, -1]) / 255.0  
        emotion_prediction = emotion_model.predict(emotion_input)
        emotion = class_names[np.argmax(emotion_prediction)]

        if age < 13 or age > 60:
            # Display red rectangle and "Not Allowed" message
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, "Not Allowed", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # Display green rectangle with age & emotion
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f'Age: {age}', (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save age and emotion to CSV
            df = pd.read_csv(csv_file)
            new_entry = pd.DataFrame({"Age": [age], "Emotion": [emotion]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(csv_file, index=False)

    cv.imshow('Age & Emotion Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
