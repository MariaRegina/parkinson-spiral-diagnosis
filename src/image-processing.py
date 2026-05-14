import cv2
import albumentations as A
import os

transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
])

transformS = A.Compose([
    A.GaussNoise(p=0.8)
])

for root, dirs, files in os.walk('../database'):
    for file in files:
        caminho = os.path.join(root, file)
        imagem = cv2.imread(caminho)

        if imagem is None:
            print(f"Erro ao carregar: {caminho}")
            continue

        augmented = transform(image=imagem)["image"]
        augmentedA = transformS(image=imagem)["image"]
        nome = os.path.splitext(file)[0]
        cv2.imwrite(f"{nome}_aug_0.jpg", augmented)
        cv2.imwrite(f"{nome}_aug_1.jpg", augmentedA)
