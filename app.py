from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Cargar modelos
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("emotion_model.h5")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    return faceNet.forward()

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotions(frame, detections, confidence_threshold=0.5):
    locs = []
    preds = []

    (h, w) = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            Xi, Yi = max(0, Xi), max(0, Yi)
            Xf, Yf = min(w, Xf), min(h, Yf)

            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue

            face = preprocess_face(face)
            pred = emotionModel.predict(face)[0]

            locs.append((Xi, Yi, Xf, Yf))
            preds.append(pred)

    return locs, preds

def draw_results(frame, locs, preds):
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)
        cv2.putText(frame, label, (Xi, Yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@socketio.on('process_frame')
def handle_process_frame(data):
    image_data = base64.b64decode(data['frame'])
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)

    # Corregir orientación invertida
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    detections = detect_faces(frame)
    locs, preds = predict_emotions(frame, detections)
    frame = draw_results(frame, locs, preds)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    emit('frame_processed', {'frame': frame_base64})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
