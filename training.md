# 🧠️ Treinamento

As imagens foram organizadas em duas classes: controle e pacientes com Doença de Parkinson.
A divisão do conjunto de dados foi realizada utilizando a biblioteca splitfolders do python ([Divisão](src/split-data.py)). 

O dataset foi separado da seguinte forma:

* 70% para treinamento
* 20% para teste
* 10% para validação

---
### 🔄 Transformações:

Para aumentar a variedade do conjunto de dados e melhorar a capacidade de generalização da rede neural, foram aplicadas técnicas de data augmentation, gerando imagens sintéticas a partir das imagens originais.

As transformações utilizadas foram:

1. Redimensionamento das imagens para 224x224 pixels
2. Rotação aleatória entre -15° e +15°
3. Conversão das imagens para tensor
4. Normalização

---

### 🧠️ Rede
Foi realizado um teste utilizando uma rede neural do tipo **Vision Transformer (ViT)**, para isso foi utilizado a
biblioteca _timm_.
O modelo foi configurado com:

* 2 neurônios de saída, correspondentes às classes do problema
* 20 épocas de treinamento

A atualização dos pesos da rede foi realizada utilizando o otimizador AdamW.


---
### 📉 Função de Perda

O erro da rede é calculado utilizando a função de perda Entropia Cruzada (Cross Entropy Loss).
Essa métrica mede a diferença entre a classificação prevista pela rede e a classe correta da imagem.

De forma geral, quanto menor o valor da perda, melhor o desempenho do modelo durante o treinamento.


---
### 📊 Resultados


|        Acurácia         |          Perda          |
|:-----------------------:|:-----------------------:|
| <img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/d2470257-b45a-478e-b92e-168e3582f56c" /> | <img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/d803d3be-1c86-45d4-83a2-96917bf9eeb3" /> |


|        | Precision | Recall | F1-Score | Support |
|--------------|------------|---------|-----------|----------|
| Control      | 1.00       | 0.78    | 0.88      | 27       |
| Parkinson    | 0.88       | 1.00    | 0.93      | 42       |
| **Accuracy** | -          | -       | **0.91**  | 69       |
| Macro Avg    | 0.94       | 0.89    | 0.90      | 69       |
| Weighted Avg | 0.92       | 0.91    | 0.91      | 69       |
