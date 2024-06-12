import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)

# Cargamos los modelos
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("modelFEC.h5")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    return faceNet.forward()

def predict_emotions(frame, detections, confidence_threshold=0.5):
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            Xi = max(0, Xi)
            Yi = max(0, Yi)
            Xf = min(frame.shape[1], Xf)
            Yf = min(frame.shape[0], Yf)

            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue

            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

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

def generate_frames():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("No se pudo abrir la cámara.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            detections = detect_faces(frame)
            locs, preds = predict_emotions(frame, detections)
            frame = draw_results(frame, locs, preds)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error en generate_frames: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
