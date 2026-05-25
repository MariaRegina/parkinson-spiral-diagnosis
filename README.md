# Image Data Increase

Recentemente foi observado que a avaliação de desenhos de espirais realizados por pacientes com Parkinson
constitui uma abordagem eficiente, de baixo custo e independente do nível de escolaridade do paciente para 
auxiliar no diagnóstico da doença. 

---

## 🎯 Objetivo:

Realizar manipulação de imagens para que a base de dados seja aumentada melhorando a eficiencia do treinamento de CNNs.

---

### 📊 Bases de Dados para Diagnóstico de Parkinson por Desenho de Espirais

As imagens foram obtidas das bases públicas:

| Dataset                                          | Referência | Controles | Parkinson | Total |
|:-------------------------------------------------|:--|----------:|----------:|--:|
| **HandPD**                                       | Pereira et al. (2016a) |        18 |        74 | 92 |
| **NewHandPD**                                    | Pereira et al. (2016b) |        35 |        31 | 66 |
| **Parkinson's Drawings**                         | Kmader (2017) |       102 |       102 | 204 |
|                                                  |  |       185 |       294 | 479 |

Foram desconsideradas bases que montaram as imagens a partir de dados de coordenadas obtidas de canetas, tablets além 
de imagens de desenho de ondas.

---

Os detalhes da realização do trabalho pode ser encontrados em:
1. [Processamento de imagens](image-processing.md)
2. [Treinamento](training.md)


---

### 📚 Referências

### HandPD
- PEREIRA, C. R. et al.  
  *A New Computer Vision-based Approach to Aid the Diagnosis of Parkinson's Disease*.  
  Computer Methods and Programs in Biomedicine, v. 136, p. 79–88, 2016.
  Disponível em: https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/

### NewHandPD
- PEREIRA, C. R. et al.  
  *Deep Learning-aided Parkinson's Disease Diagnosis from Handwritten Dynamics*.  
  Proceedings of SIBGRAPI 2016, p. 340–346, 2016.
  Disponível em: https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/

### Parkinson's Drawings
- KMADER.  
  *Parkinson's Drawings*.  
  Kaggle, 2017.  
  Disponível em: https://www.kaggle.com/datasets/kmader/parkinsons-drawings
