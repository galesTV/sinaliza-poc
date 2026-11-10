import csv
from pathlib import Path

import numpy as np


DATASET_PATH = Path(__file__).parent / "dataset.csv"

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
    """
    Lê o CSV e organiza as características por configuração.
    """
    dataset = {}

    with open(DATASET_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["rotulo"]

            features = [
                float(row[feature_name])
                for feature_name in FEATURE_NAMES
            ]

            if label not in dataset:
                dataset[label] = []

            dataset[label].append(features)

    return dataset


def analyze_dataset(dataset):
    """
    Calcula estatísticas para cada configuração.
    """
    for label, samples in dataset.items():
        data = np.array(samples, dtype=np.float32)

        print("\n" + "=" * 65)
        print(f"CONFIGURAÇÃO: {label.upper()}")
        print(f"Quantidade de amostras: {len(data)}")
        print("=" * 65)

        print(
            f"{'CARACTERÍSTICA':<32}"
            f"{'MÉDIA':>10}"
            f"{'MÍNIMO':>10}"
            f"{'MÁXIMO':>10}"
            f"{'DESVIO':>10}"
        )

        for index, feature_name in enumerate(FEATURE_NAMES):
            values = data[:, index]

            mean = np.mean(values)
            minimum = np.min(values)
            maximum = np.max(values)
            std = np.std(values)

            print(
                f"{feature_name:<32}"
                f"{mean:>10.2f}"
                f"{minimum:>10.2f}"
                f"{maximum:>10.2f}"
                f"{std:>10.2f}"
            )


def main():
    if not DATASET_PATH.exists():
        print("Erro: dataset.csv não encontrado.")
        return

    dataset = load_dataset()

    if not dataset:
        print("O dataset está vazio.")
        return

    total_samples = sum(
        len(samples)
        for samples in dataset.values()
    )

    print("=" * 65)
    print("ANÁLISE DO DATASET - SINALIZA")
    print("=" * 65)
    print(f"Total de amostras: {total_samples}")

    analyze_dataset(dataset)

    print("\n" + "=" * 65)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    main()
