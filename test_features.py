
import numpy as np

from feature_extractor import (
    calculate_distance,
    calculate_angle,
    extract_features
)


# Simula uma mão com 21 landmarks
fake_landmarks = np.zeros((21, 3), dtype=np.float32)

# Define algumas posições fictícias
# Polegar
fake_landmarks[2] = [0.0, 0.0, 0.0]
fake_landmarks[3] = [0.0, 1.0, 0.0]
fake_landmarks[4] = [1.0, 1.0, 0.0]

# Dedo indicador
fake_landmarks[4] = [1.0, 0.0, 0.0]
fake_landmarks[5] = [1.0, 1.0, 0.0]
fake_landmarks[6] = [2.0, 1.0, 0.0]
fake_landmarks[8] = [2.0, 2.0, 0.0]

# Dedo médio
fake_landmarks[9] = [3.0, 0.0, 0.0]
fake_landmarks[10] = [3.0, 1.0, 0.0]
fake_landmarks[12] = [4.0, 1.0, 0.0]

# Dedo anelar
fake_landmarks[13] = [5.0, 0.0, 0.0]
fake_landmarks[14] = [5.0, 1.0, 0.0]
fake_landmarks[16] = [6.0, 1.0, 0.0]

# Dedo mínimo
fake_landmarks[17] = [7.0, 0.0, 0.0]
fake_landmarks[18] = [7.0, 1.0, 0.0]
fake_landmarks[20] = [8.0, 1.0, 0.0]

features = extract_features(fake_landmarks)

print("Vetor de características:", features)
print("Formato do vetor:", features.shape)
print("Quantidade de características:", len(features))
