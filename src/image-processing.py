import cv2
import albumentations as A
import os
import numpy as np

transform = A.Compose([
    A.Affine(
        rotate=(-45,45),
        fill=(255,255,255),
        border_mode=cv2.BORDER_CONSTANT,
        interpolation=cv2.INTER_LANCZOS4,
        p=1.0,
        fit_output=True
    )
])

transformS = A.Compose([
    A.GaussNoise(p=0.8)
])

for root, dirs, files in os.walk('../database'):
    index = 1
    for file in files:
        caminho = os.path.join(root, file)
        imagem = cv2.imread(caminho)

        if imagem is None:
            print(f"Erro ao carregar: {caminho}")
            continue

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])

        augmented = transform(image=imagem)["image"]

        hsvOriginal = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)

        maskOriginal = cv2.inRange(hsvOriginal, lower_blue, upper_blue)

        original = np.ones_like(imagem) * 255

        original[maskOriginal> 0] = [255, 0, 0]

        augmented = transform(image=original)["image"]
        augmentedA = transformS(image=original)["image"]

        typeImage = ''
        if 'Circle' in caminho:
            typeImage = 'Circle/'
        else:
            if 'Meander' in caminho:
                typeImage = 'Meander/'
            else:
                typeImage = 'Spiral/'

        if 'Control' in caminho:
            nome = '../processed-images/Control/' + typeImage
        else:
            nome = '../processed-images/Parkinson/' + typeImage

        cv2.imwrite(nome + str(index) + "-rotated.jpg", augmented)
        cv2.imwrite(nome + str(index) + "-noise.jpg", augmentedA)
        cv2.imwrite(nome + str(index) + "-original.jpg", original)

        index += 1
        # break

print('Processamento finalizado')