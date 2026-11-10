
import numpy as np


def extract_landmarks(hand_landmarks, image_width, image_height):
    """
    Recebe os 21 landmarks do MediaPipe e converte
    suas coordenadas para uma escala baseada em pixels.

    Retorna uma matriz NumPy de formato (21, 3).
    """

    points = []

    for landmark in hand_landmarks:
        points.append([
            landmark.x * image_width,
            landmark.y * image_height,
            landmark.z * image_width
        ])

    return np.array(points, dtype=np.float32)

def normalize_landmarks(landmarks):
    """
    Normaliza os 21 landmarks utilizando o pulso
    como origem e a distância entre o pulso e
    a base do dedo médio como referência de escala.
    """

    # Obtém o landmark do pulso (índice 0)
    wrist = landmarks[0]

    # Transforma todas as coordenadas em posições relativas
    relative_landmarks = landmarks - wrist

    # Obtém a distância entre o pulso e a base do dedo médio
    reference_distance = np.linalg.norm(
        relative_landmarks[9, :2]
    )

    # Evita divisão por zero
    if reference_distance < 1e-6:
        raise ValueError("Distância de referência muito pequena.")

    # Ajusta a escala de todos os landmarks
    normalized_landmarks = (
        relative_landmarks / reference_distance
    )

    return normalized_landmarks

def calculate_distance(point_a, point_b):
    """
    Calcula a distância euclidiana entre dois landmarks
    em três dimensões (x, y, z).
    """

    distance = np.linalg.norm(point_a - point_b)

    return float(distance)

def calculate_angle(point_a, point_b, point_c):
    """
    Calcula o ângulo em graus formado por três pontos.
    O ponto B é o vértice do ângulo.
    """

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    dot_product = np.dot(vector_ba, vector_bc)

    magnitude_ba = np.linalg.norm(vector_ba)
    magnitude_bc = np.linalg.norm(vector_bc)

    if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
        raise ValueError("Não é possível calcular o ângulo.")

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return float(angle)


def extract_features(normalized_landmarks):
    """
    Extrai características geométricas da mão.

    Retorna:
    - Distância polegar-indicador
    - Distância pulso-indicador
    - Ângulo do indicador
    - Ângulo do dedo médio
    - Ângulo do anelar
    - Ângulo do dedo mínimo
    """

    # Distância entre polegar e indicador
    thumb_index_distance = calculate_distance(
        normalized_landmarks[4],
        normalized_landmarks[8]
    )

    # Distância entre pulso e ponta do indicador
    wrist_index_distance = calculate_distance(
        normalized_landmarks[0],
        normalized_landmarks[8]
    )
    
    # Ângulo do polegar
    thumb_angle = calculate_angle(
        normalized_landmarks[2],
        normalized_landmarks[3],
        normalized_landmarks[4]
    )

    # Ângulo do indicador
    index_angle = calculate_angle(
        normalized_landmarks[5],
        normalized_landmarks[6],
        normalized_landmarks[8]
    )

    # Ângulo do dedo médio
    middle_angle = calculate_angle(
        normalized_landmarks[9],
        normalized_landmarks[10],
        normalized_landmarks[12]
    )

    # Ângulo do dedo anelar
    ring_angle = calculate_angle(
        normalized_landmarks[13],
        normalized_landmarks[14],
        normalized_landmarks[16]
    )

    # Ângulo do dedo mínimo
    pinky_angle = calculate_angle(
        normalized_landmarks[17],
        normalized_landmarks[18],
        normalized_landmarks[20]
    )

    # Junta todas as características
    features = np.array([
        thumb_index_distance,
        wrist_index_distance,
        thumb_angle,
        index_angle,
        middle_angle,
        ring_angle,
        pinky_angle
    ], dtype=np.float32)

    return features

