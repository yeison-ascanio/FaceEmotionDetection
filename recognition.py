import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from PIL import Image, ImageTk

# Cargamos los modelos
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("modelFEC.h5")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionRecognitionApp:
    def __init__(self, root, show_welcome_callback):
        self.root = root
        self.root.title("Detección de Emociones")
        self.show_welcome_callback = show_welcome_callback

        # Crear una etiqueta para mostrar la cámara
        self.label = tk.Label(root)
        self.label.pack()

        # Crear un botón para regresar a la ventana de bienvenida
        self.back_button = tk.Button(root, text="Regresar", command=self.go_back)
        self.back_button.pack(pady=20)

        # Iniciar captura de video
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara.")

        self.update_frame()

    def go_back(self):
        self.cap.release()
        self.root.destroy()
        self.show_welcome_callback()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma de la cámara.")
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        detections = self.detect_faces(frame)
        locs, preds = self.predict_emotions(frame, detections)
        frame = self.draw_results(frame, locs, preds)

        # Convertir el frame a un formato que Tkinter puede manejar
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        self.label.config(image=image)
        self.label.image = image

        self.root.after(10, self.update_frame)

    def detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        return faceNet.forward()

    def predict_emotions(self, frame, detections, confidence_threshold=0.5):
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

    def __init__(self, root, show_welcome_callback):
        self.root = root
        self.root.title("Detección de Emociones")
        self.show_welcome_callback = show_welcome_callback

        # Crear una etiqueta para mostrar la cámara
        self.label = tk.Label(root)
        self.label.pack()

        # Crear un botón para regresar a la ventana de bienvenida
        self.back_button = tk.Button(root, text="Regresar", command=self.go_back)
        self.back_button.pack(pady=20)

        # Iniciar captura de video
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara.")

        self.update_frame()

    def go_back(self):
        self.cap.release()
        self.root.destroy()
        self.show_welcome_callback()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma de la cámara.")
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        detections = self.detect_faces(frame)
        locs, preds = self.predict_emotions(frame, detections)
        frame = self.draw_results(frame, locs, preds)

        # Convertir el frame a un formato que Tkinter puede manejar
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        self.label.config(image=image)
        self.label.image = image

        self.root.after(10, self.update_frame)

    def detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        return faceNet.forward()

    def predict_emotions(self, frame, detections, confidence_threshold=0.5):
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

    def draw_results(self, frame, locs, preds):
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)
            cv2.putText(frame, label, (Xi, Yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        return frame


