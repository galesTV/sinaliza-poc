
import csv
import json
from pathlib import Path


CSV_PATH = Path(__file__).parent / "dataset.csv"
JSON_PATH = Path(__file__).parent / "dataset.json"


FEATURE_NAMES = [
    "distancia_polegar_indicador",
    "distancia_pulso_indicador",
    "angulo_indicador",
    "angulo_medio",
    "angulo_anelar",
    "angulo_minimo",
    "angulo_polegar",
]


def convert_dataset():
    dataset = {}

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["rotulo"]

            sample = {
                feature: float(row[feature])
                for feature in FEATURE_NAMES
            }

            if label not in dataset:
                dataset[label] = []

            dataset[label].append(sample)

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    return dataset


def main():
    if not CSV_PATH.exists():
        print("Erro: dataset.csv não encontrado.")
        return

    dataset = convert_dataset()

    total = sum(len(samples) for samples in dataset.values())

    print("Conversão concluída!")
    print(f"Arquivo criado: {JSON_PATH.name}")
    print(f"Total de amostras: {total}")

    for label, samples in dataset.items():
        print(f"{label}: {len(samples)} amostras")


if __name__ == "__main__":
    main()
