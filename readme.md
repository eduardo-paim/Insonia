# Projeto de Classificação de Imagens com ResNet50 e TensorFlow

Este projeto utiliza TensorFlow e a arquitetura de rede neural ResNet50 para criar um modelo de classificação de imagens eficiente e robusto, aplicado ao conjunto de dados CIFAR-10.

## Visão Geral do Projeto
O objetivo deste projeto é construir e treinar um modelo de rede neural convolucional (CNN) para reconhecer e categorizar imagens do conjunto de dados CIFAR-10, que inclui 60.000 imagens em 10 classes diferentes.

## Pré-requisitos
- Python 3.6 ou superior
- Bibliotecas TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn

## Instalação
Para instalar as bibliotecas necessárias, execute:
pip install tensorflow numpy matplotlib seaborn scikit-learn

## Estrutura do Código
O código está estruturado da seguinte forma:
- Carregamento e pré-processamento do conjunto de dados CIFAR-10
- Construção do modelo utilizando a arquitetura ResNet50
- Compilação e treinamento do modelo
- Fine-tuning de algumas camadas da ResNet50
- Avaliação do modelo usando o conjunto de teste
- Geração de matrizes de confusão e relatórios de classificação para análise do desempenho

## Características do Modelo
- **Arquitetura de Rede**: ResNet50
- **Regularização**: Dropout e Regularização L2
- **Função de Perda**: Sparse Categorical Crossentropy

## Treinamento do Modelo
O modelo é inicialmente treinado com as camadas da ResNet50 congeladas, seguido por um processo de fine-tuning nas últimas camadas para melhor ajuste aos dados CIFAR-10.

## Avaliação e Análise
- **Matriz de Confusão**: Visualização da performance do modelo em classificar cada classe.
- **Relatório de Classificação**: Métricas detalhadas como precisão, recall e F1-scor
