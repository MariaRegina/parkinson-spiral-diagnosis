import cv2
import albumentations as A
import os
import numpy as np

# transform = A.Compose([
#     A.Affine(
#         rotate=(-45,45),
#         fill=(255,255,255),
#         border_mode=cv2.BORDER_CONSTANT,
#         interpolation=cv2.INTER_LANCZOS4,
#         p=1.0,
#         fit_output=True
#     )
# ])
#
# transformS = A.Compose([
#     A.GaussNoise(p=0.8)
# ])

for root, dirs, files in os.walk('../database'):
    index = 1
    for file in files:
        caminho = os.path.join(root, file)
        imagem = cv2.imread(caminho)

        if imagem is None:
            print(f"Erro ao carregar: {caminho}")
            continue

        typeImage = ''

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 200])

        # augmented = transform(image=imagem)["image"]

        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        maskOriginal = cv2.inRange(hsv, lower_blue, upper_blue)
        maskBlackOriginal = cv2.inRange(hsv, lower_black, upper_black)

        original = np.ones_like(imagem) * 255
        originalBlack = np.ones_like(imagem) * 255

        original[maskOriginal> 0] = [255, 0, 0]
        originalBlack[maskBlackOriginal > 0] = [255, 0, 0]

        if 'Circle' in caminho:
            typeImage = 'Circle/'
        else:
            if 'Meander' in caminho:
                typeImage = 'Meander/'
            else:
                if 'Lapis' in caminho:
                    typeImage = 'Lapis/l-'
                else:
                    typeImage = 'Spiral/'



        if 'Control' in caminho:
            nome = f'../processed-images/Control/{typeImage}' + 'control-'
        else:
            nome = f'../processed-images/Parkinson/{typeImage}' + 'parkinson-'

        cv2.imwrite(nome + str(index) + "-original.jpg", original)
        # cv2.imwrite(nome + str(index) + "-original.jpg", originalBlack) #para os casos de espirais desenhadas com lapis

        index += 1

print('Processamento finalizado')