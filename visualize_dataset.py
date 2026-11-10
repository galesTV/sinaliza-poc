
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET_PATH = Path(__file__).parent / "dataset.csv"


def load_dataset():
    """
    Carrega as características e separa por configuração.
    """
    dataset = {}

    with open(DATASET_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["rotulo"]

            features = [
                float(row["distancia_polegar_indicador"]),
                float(row["distancia_pulso_indicador"]),
                float(row["angulo_indicador"]),
                float(row["angulo_medio"]),
                float(row["angulo_anelar"]),
                float(row["angulo_minimo"]),
                float(row["angulo_polegar"]),
            ]

            if label not in dataset:
                dataset[label] = []

            dataset[label].append(features)

    return dataset


def plot_features(dataset):
    """
    Cria um gráfico comparando distância pulso-indicador
    e ângulo do indicador.
    """
    plt.figure(figsize=(10, 6))

    for label, samples in dataset.items():
        data = np.array(samples)

        x = data[:, 1]
        y = data[:, 2]

        plt.scatter(
            x,
            y,
            label=label,
            alpha=0.7
        )

    plt.title("Distribuição das configurações de mão")
    plt.xlabel("Distância entre pulso e indicador")
    plt.ylabel("Ângulo do indicador (graus)")

    plt.legend(title="Configuração")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    if not DATASET_PATH.exists():
        print("Erro: dataset.csv não encontrado.")
        return

    dataset = load_dataset()

    if not dataset:
        print("O dataset está vazio.")
        return

    plot_features(dataset)


if __name__ == "__main__":
    main()
