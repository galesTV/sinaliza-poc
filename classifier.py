
import json
from pathlib import Path

import numpy as np


DATASET_PATH = Path(__file__).parent / "dataset.json"

FEATURE_NAMES = [
    "distancia_polegar_indicador",
    "distancia_pulso_indicador",
    "angulo_indicador",
    "angulo_medio",
    "angulo_anelar",
    "angulo_minimo",
    "angulo_polegar",
]


def load_dataset():
    """Carrega as amostras cadastradas no JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_model(dataset):
    """
    Prepara as amostras para comparação.

    Padroniza cada característica usando média e
    desvio padrão do conjunto de treinamento.
    """
    samples = []
    labels = []

    for label, class_samples in dataset.items():
        for sample in class_samples:
            features = [
                sample[name]
                for name in FEATURE_NAMES
            ]

            samples.append(features)
            labels.append(label)

    matrix = np.array(samples, dtype=np.float32)

    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    # Evita divisão por zero.
    std[std < 1e-8] = 1.0

    standardized_samples = (matrix - mean) / std

    return {
        "samples": standardized_samples,
        "labels": labels,
        "mean": mean,
        "std": std,
    }


def classify(features, model):
    """
    Encontra a amostra cadastrada mais próxima
    usando distância euclidiana.
    """
    features = np.array(features, dtype=np.float32)

    if features.shape != (len(FEATURE_NAMES),):
        raise ValueError("Esperadas exatamente 7 características.")

    standardized_features = (
        features - model["mean"]
    ) / model["std"]

    differences = (
        model["samples"] - standardized_features
    )

    distances = np.linalg.norm(differences, axis=1)

    nearest_index = np.argmin(distances)

    predicted_label = model["labels"][nearest_index]
    nearest_distance = float(distances[nearest_index])

    return predicted_label, nearest_distance


def evaluate_leave_one_out(dataset):
    """
    Avalia cada amostra usando as demais como referência.

    Cada amostra é retirada temporariamente do treinamento
    antes de ser classificada.
    """
    correct = 0
    total = 0

    confusion = {
        label: {
            predicted: 0
            for predicted in dataset
        }
        for label in dataset
    }

    for actual_label, samples in dataset.items():
        for index, sample in enumerate(samples):
            training_dataset = {}

            for label, class_samples in dataset.items():
                remaining = [
                    item
                    for i, item in enumerate(class_samples)
                    if not (
                        label == actual_label and i == index
                    )
                ]

                if remaining:
                    training_dataset[label] = remaining

            model = build_model(training_dataset)

            features = [
                sample[name]
                for name in FEATURE_NAMES
            ]

            predicted_label, _ = classify(features, model)

            confusion[actual_label][predicted_label] += 1

            if predicted_label == actual_label:
                correct += 1

            total += 1

    return correct, total, confusion


def main():
    dataset = load_dataset()

    model = build_model(dataset)

    print("MODELO PREPARADO")
    print(f"Amostras cadastradas: {len(model['labels'])}")
    print(f"Configurações: {', '.join(dataset.keys())}")

    correct, total, confusion = evaluate_leave_one_out(dataset)

    accuracy = (correct / total) * 100

    print("\nVALIDAÇÃO LEAVE-ONE-OUT")
    print(f"Classificações corretas: {correct}/{total}")
    print(f"Acurácia observada: {accuracy:.2f}%")

    print("\nMATRIZ DE CONFUSÃO")
    print(f"{'REAL / PREVISTO':<18}", end="")

    for label in dataset:
        print(f"{label:>12}", end="")

    print()

    for actual_label, predictions in confusion.items():
        print(f"{actual_label:<18}", end="")

        for predicted_label in dataset:
            print(f"{predictions[predicted_label]:>12}", end="")

        print()


if __name__ == "__main__":
    main()
