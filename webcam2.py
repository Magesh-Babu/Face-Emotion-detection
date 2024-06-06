import sys
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the face detection model
face_classifier = cv2.CascadeClassifier(r'C:\Users\dines\Downloads\Emotion_Detection_CNN-main\models\haarcascade_frontalface_default.xml')

if face_classifier.empty():
    print("Error: Could not load face cascade classifier.")
    sys.exit()

# Load the emotion detection model
try:
    classifier = load_model(r'C:\Users\dines\Downloads\Emotion_Detection_CNN-main\models\model_28_.h5', compile=False)
    print("Model loaded successfully.")
    
    # Recompile the model with a new optimizer
    from keras.optimizers import Adam
    classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error: Could not load model - {e}")
    sys.exit()

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=-1)  # Ensure the shape is (224, 224, 1)
            roi = np.expand_dims(roi, axis=0)  # Shape becomes (1, 224, 224, 1)

            # Make a prediction on the ROI
            prediction = classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
