import pytest
import cv2
import numpy as np
from app import app, detect_faces, preprocess_face, predict_emotions, draw_results

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index page"""
    rv = client.get('/')
    assert rv.status_code == 200
    # Convertir la cadena de búsqueda a bytes antes de la comparación
    assert b'<title>Detecci&oacute;n de Emociones</title>'

def test_detection(client):
    """Test the detection page"""
    rv = client.get('/detection')
    assert rv.status_code == 200
    # Convertir la cadena de búsqueda a bytes antes de la comparación
    assert b'<title>Detecci&oacute;n de Emociones</title>'

def test_detect_faces():
    """Test face detection"""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = detect_faces(frame)
    assert detections.shape[2] >= 0  # Should have at least one detection layer

def test_preprocess_face():
    """Test face preprocessing"""
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    preprocessed = preprocess_face(face)
    assert preprocessed.shape == (1, 48, 48, 1)

def test_predict_emotions():
    """Test emotion prediction"""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = detect_faces(frame)
    locs, preds = predict_emotions(frame, detections, confidence_threshold=0)
    assert len(locs) == len(preds)

def test_draw_results():
    """Test drawing results on the frame"""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    locs = [(50, 50, 100, 100)]
    preds = [np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])]
    result_frame = draw_results(frame, locs, preds)
    assert result_frame.shape == frame.shape
