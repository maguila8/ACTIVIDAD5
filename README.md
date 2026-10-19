# Clasificación de mensajes de atención al cliente

## Descripción del proyecto

Este proyecto consiste en preparar un dataset público de mensajes de atención al cliente y comparar dos modelos de clasificación de texto: Regresión Logística y Naive Bayes.

Se utiliza el texto de los mensajes para predecir su categoría. Los datos se revisan y limpian antes del entrenamiento. Después, se aplica Grid Search con validación cruzada para buscar los mejores parámetros de cada modelo.

También se utiliza MLflow para registrar los experimentos, sus parámetros, métricas y modelos entrenados.

## Dataset

Se utiliza el dataset público [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), disponible en Hugging Face.

El dataset contiene mensajes de atención al cliente organizados en diferentes categorías. Para este proyecto:

- **Variable de entrada:** `instruction`, que contiene el mensaje del cliente.
- **Variable objetivo:** `category`, que indica la categoría que se busca predecir.

Se conserva una copia del dataset original y otra del dataset limpio para facilitar la revisión y reproducibilidad del proceso.

## Modelos utilizados

- **Regresión Logística:** modelo de clasificación utilizado como primera alternativa.
- **Naive Bayes:** modelo probabilístico utilizado como segunda alternativa.

Ambos modelos utilizan TF-IDF para representar los textos numéricamente. Se comparan mediante accuracy, precision, recall y F1-score, además del tiempo de búsqueda de parámetros.

## Estructura del repositorio

- `datos/datos_ini/`: dataset original.
- `datos/datos_limp/`: dataset limpio.
- `fuentes/entrena.ipynb`: notebook con el desarrollo del proyecto.
- `fuentes/datos_prep.py`: funciones para preparar y
