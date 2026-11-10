
import numpy as np


# Coordenadas fictícias de uma mão
# Cada linha representa um ponto (X, Y)

wrist = np.array([0.40, 0.60])

index_tip = np.array([0.55, 0.35])

middle_base = np.array([0.50, 0.45])


# 1. Coordenada original
print("Coordenada original:")
print(index_tip)


# 2. Coordenada relativa ao pulso
relative_position = index_tip - wrist

print("\nCoordenada relativa ao pulso:")
print(relative_position)


# 3. Distância entre pulso e base do dedo médio
reference_distance = np.linalg.norm(
    middle_base - wrist
)

print("\nDistância de referência:")
print(reference_distance)


# 4. Normalização pela distância de referência
normalized_position = (
    relative_position / reference_distance
)

print("\nCoordenada normalizada:")
print(normalized_position)
