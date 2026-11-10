
import cv2
import mediapipe as mp
import time
import numpy as np
import json

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import (
    extract_landmarks,
    normalize_landmarks,
    extract_features
)

from classifier import (
    load_dataset,
    build_model,
    classify
)


MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset.json"

FEATURE_NAMES = [
    "distancia_polegar_indicador",
    "distancia_pulso_indicador",
    "angulo_indicador",
    "angulo_medio",
    "angulo_anelar",
    "angulo_minimo",
    "angulo_polegar"
]

LABELS = {
    ord("1"): "aberta",
    ord("2"): "punho",
    ord("3"): "pinca"
}


def save_sample(label, features):
    """
    Salva uma nova amostra no dataset.json.
    Preserva todas as amostras já cadastradas.
    """

    dataset_path = Path(DATASET_PATH)

    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)
    else:
        dataset = {}

    if label not in dataset:
        dataset[label] = []

    sample = {
        feature_name: float(value)
        for feature_name, value in zip(FEATURE_NAMES, features)
    }

    dataset[label].append(sample)

    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    print(
        f"Amostra salva: {label} "
        f"({len(dataset[label])} amostras cadastradas)"
    )


def main():
    # Carrega o dataset e prepara o classificador uma única vez.
    dataset = load_dataset()
    model = build_model(dataset)

    print("MODELO CARREGADO")
    print(f"Amostras cadastradas: {len(model['labels'])}")
    print("Identificacao automatica ativada!")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )

    selected_label = None
    last_timestamp = 0

    with vision.HandLandmarker.create_from_options(options) as detector:

        print("\nSinaliza iniciado!")
        print("IDENTIFICACAO AUTOMATICA")
        print("1 - Selecionar mao aberta para cadastro")
        print("2 - Selecionar punho para cadastro")
        print("3 - Selecionar pinca para cadastro")
        print("S - Salvar amostra")
        print("Q - Sair")

        while True:
            success, frame = camera.read()

            if not success:
                print("Erro ao capturar imagem.")
                break

            height, width, _ = frame.shape

            current_features = None
            predicted_label = None
            prediction_distance = None

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = max(
                last_timestamp + 1,
                time.monotonic_ns() // 1_000_000
            )

            last_timestamp = timestamp

            result = detector.detect_for_video(
                mp_image,
                timestamp
            )

            if result.hand_landmarks:

                for hand_landmarks in result.hand_landmarks:

                    landmarks_array = extract_landmarks(
                        hand_landmarks,
                        width,
                        height
                    )

                    normalized_array = normalize_landmarks(
                        landmarks_array
                    )

                    features = extract_features(
                        normalized_array
                    )

                    current_features = features

                    # Classifica automaticamente a configuração.
                    predicted_label, prediction_distance = classify(
                        current_features,
                        model
                    )

                    # Painel das características.
                    panel_width = 390
                    panel_height = 205

                    panel_x = width - panel_width - 10
                    panel_y = 10

                    cv2.rectangle(
                        frame,
                        (panel_x, panel_y),
                        (panel_x + panel_width, panel_y + panel_height),
                        (0, 0, 0),
                        -1
                    )

                    for index, (name, value) in enumerate(
                        zip(FEATURE_NAMES, features)
                    ):
                        text = f"{name}: {value:.2f}"

                        cv2.putText(
                            frame,
                            text,
                            (panel_x + 10, panel_y + 25 + index * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

                    # Desenha as conexões entre os landmarks.
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17)
                    ]

                    for start, end in connections:
                        point_a = hand_landmarks[start]
                        point_b = hand_landmarks[end]

                        x1 = int(point_a.x * width)
                        y1 = int(point_a.y * height)

                        x2 = int(point_b.x * width)
                        y2 = int(point_b.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    # Desenha os 21 landmarks.
                    for index, landmark in enumerate(hand_landmarks):
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            frame,
                            str(index),
                            (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 0),
                            1
                        )

                cv2.putText(
                    frame,
                    "Mao detectada - Normalizacao ativa",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # Resultado da classificação.
                cv2.putText(
                    frame,
                    f"Identificacao: {predicted_label}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Distancia: {prediction_distance:.3f}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            else:
                cv2.putText(
                    frame,
                    "Nenhuma mao detectada",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Identificacao: aguardando...",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            # Exibe a configuração selecionada para cadastro.
            registration_text = (
                f"Cadastro selecionado: {selected_label}"
                if selected_label
                else "Cadastro selecionado: nenhum"
            )

            cv2.putText(
                frame,
                registration_text,
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # Instruções na tela.
            cv2.putText(
                frame,
                "1: Aberta | 2: Punho | 3: Pinca",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "S: Salvar amostra | Q: Sair",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("Sinaliza - Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Seleciona uma configuração para cadastro.
            if key in LABELS:
                selected_label = LABELS[key]
                print(f"Configuracao selecionada: {selected_label}")

            # Salva uma nova amostra e atualiza o modelo.
            elif key == ord("s"):

                if selected_label is None:
                    print("Selecione uma configuracao primeiro!")

                elif current_features is None:
                    print("Nenhuma mao detectada. Amostra nao salva.")

                else:
                    save_sample(
                        selected_label,
                        current_features
                    )

                    # Reconstrói o modelo incluindo a nova amostra.
                    model = build_model(load_dataset())

                    print(
                        f"Modelo atualizado! "
                        f"Total de amostras: {len(model['labels'])}"
                    )

            elif key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
