# opencv-project

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('captura.png')

# Apresentar a imagem original
plt.grid(False)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.show()


# Calcular o histograma para cada canal de cor
# Canais: 0-Azul, 1-Verde, 2-Vermelho
azul = cv2.calcHist([imagem], [0], None, [256], [0, 256])
verde = cv2.calcHist([imagem], [1], None, [256], [0, 256])
vermelha = cv2.calcHist([imagem], [2], None, [256], [0, 256])

# Normalizar os histogramas para criar vetor de características
azul = cv2.normalize(azul, azul).flatten()
verde = cv2.normalize(verde, verde).flatten()
vermelha = cv2.normalize(vermelha, vermelha).flatten()

# Concatenar os vetores dos três canais em um único vetor de características
vetor_caracteristicas = np.concatenate([vermelha, verde, azul])


# Exibir os histogramas
plt.title("Histograma de Cores da Imagem")
plt.xlabel("Valor do Pixel")
plt.ylabel("Frequência Normalizada")
plt.hist(vermelha, color='red', label='Vermelho')
plt.hist(verde, color='green', label='Verde')
plt.hist(azul, color='blue', label='Azul')
plt.show()


# Exibir o vetor de características
print("Vetor de características (histograma colorido):")
print(vetor_caracteristicas)