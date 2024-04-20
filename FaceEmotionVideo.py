# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Cargamos el modelo de detección de rostros
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el modelo de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Función para detectar rostros y emociones en tiempo real
def detect_emotions():
    # Crea un objeto VideoCapture para acceder a la cámara
    cap = cv2.VideoCapture(0)

    # Verifica si la cámara se ha abierto correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Lee el video de la cámara en un bucle
    while True:
        # Lee un fotograma de la cámara
        ret, frame = cap.read()

        # Verifica si se pudo leer el fotograma correctamente
        if not ret:
            print("Error: No se pudo leer el fotograma de la cámara.")
            break

        # Muestra el fotograma en una ventana en modo espejo
        frame = cv2.flip(frame, 1)
        cv2.imshow("Face Emotion Detection", frame)

        # Realiza la detección de rostros y emociones
        locs, preds = predict_emotion(frame, faceNet, emotionModel)

        # Dibuja los cuadros delimitadores y etiquetas de emociones en el fotograma
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)
            cv2.putText(frame, label, (Xi, Yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Muestra el fotograma en una ventana
        cv2.imshow("Face Emotion Detection", frame)

        # Espera 1 milisegundo y verifica si se ha presionado la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la captura de la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Función para predecir emociones a partir de un fotograma
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir del blob
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar las ubicaciones de los rostros y las predicciones de emociones
    locs = []
    preds = []

    # Recorre cada una de las detecciones de rostros
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtra las detecciones débiles
        if confidence > 0.5:
            # Obtiene las coordenadas del cuadro delimitador del rostro
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del cuadro delimitador
            if Xi < 0:
                Xi = 0
            if Yi < 0:
                Yi = 0

            # Extrae el rostro de la imagen y realiza la preprocesamiento
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Realiza la predicción de la emoción
            pred = emotionModel.predict(face)[0]

            # Agrega las ubicaciones de los rostros y las predicciones de emociones a las listas
            locs.append((Xi, Yi, Xf, Yf))
            preds.append(pred)

    return locs, preds

# Llama a la función para detectar rostros y emociones en tiempo real
detect_emotions()
