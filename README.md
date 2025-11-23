# 🎵 Clasificación de Géneros Musicales con Spotify API & XGBoost

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?style=for-the-badge&logo=jupyter)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-InProgress-success?style=for-the-badge)

> **Entrega Práctica MD3 - Machine Learning**
>
> **Autor:** ChengjiePL
> **Fecha:** Noviembre 2025

---

## 📖 Descripción del Proyecto

Este proyecto aborda un problema clásico de clasificación supervisada: **Predecir el género musical de una canción (Rock, Dance, Classical, Acoustic) basándose únicamente en sus propiedades físico-matemáticas.**

Utilizando un dataset de canciones extraído de la API de Spotify, se ha desarrollado un flujo de trabajo completo de Data Science, desde el análisis exploratorio inicial hasta la optimización de modelos avanzados de Gradient Boosting.

### 🎯 Objetivo Principal

Desarrollar un modelo predictivo capaz de distinguir patrones sónicos complejos, como diferenciar una canción de rock (alta energía, instrumentación real) de una canción acústica (baja energía, instrumentación real) o una pista de baile (alta energía, sintética).

---

## 📂 Estructura del Repositorio

---

## 🧠 Metodología y Fases del Proyecto (Notebook)

El notebook `Music_Genre_Classification.ipynb` sigue una estructura rigurosa de 6 fases:

### 1. Análisis Exploratorio de Datos (EDA) 📊

Antes de modelar, se realizó una "radiografía" completa de los datos para entender qué define a cada género:
*   **Distribución de Variables:** Uso de histogramas y *boxplots* para identificar que, por ejemplo, la `danceability` es el discriminante clave entre *Classical* y *Dance*.
*   **Mapa de Correlaciones:** Detección de multicolinealidad. Se descubrió una fuerte correlación negativa entre `energy` y `acousticness`.
*   **Análisis de Outliers:** Identificación de canciones atípicas (ej: canciones de rock muy suaves) que podrían confundir al modelo.

### 2. Feature Engineering🛠️

Para mejorar la capacidad predictiva, no nos limitamos a las variables originales. Creamos nuevas métricas sintéticas basadas en conocimiento del dominio musical:
*   **`Intensity`**: Producto de `energy * loudness`. Captura la "potencia" percibile.
*   **`Dance_Tempo`**: Relación entre ritmo y velocidad.
*   **`Chill_Factor`**: Diferencia entre valencia positiva y energía, útil para separar géneros relajados.

### 3. Preprocesamiento de Datos 🧹

*   Codificación de variables categóricas (`LabelEncoder`).
*   Escalado de datos (`StandardScaler`) para algoritmos sensibles a la magnitud (como KNN en la fase experimental).
*   División estratificada del dataset (Train/Test Split) para garantizar que todos los géneros estén representados equitativamente.

### 4. Selección y Entrenamiento de Modelos 🤖

Se sometieron a prueba dos familias de algoritmos:
1.  **Random Forest:** Como modelo base de *bagging*.
2.  **XGBoost (Extreme Gradient Boosting):** Como modelo avanzado de *boosting*.

**Resultado:** XGBoost superó al Random Forest en métricas de precisión y ROC-AUC, demostrando mayor capacidad para manejar las fronteras de decisión complejas entre *Rock* y *Acoustic*.

### 5. Evaluación y Métricas 📈

El modelo final fue auditado exhaustivamente:
*   **Matriz de Confusión:** Análisis de errores tipo I y II. (ej: ¿Con qué confunde la IA al Rock?).
*   **Curva ROC / AUC:** Validación de la robustez del clasificador (>0.95 AUC).
*   **Feature Importance:** Confirmación de que `acousticness` y `loudness` son los predictores más potentes.

---

## 🏆 Resultados Clave

| Métrica | Random Forest | **XGBoost (Final)** |
| :--- | :---: | :---: |
| Accuracy | 89% | **92%** |
| F1-Score (Macro) | 0.88 | **0.91** |


> **Conclusión Técnica:** El modelo demuestra que los géneros musicales no son etiquetas subjetivas, sino clústeres matemáticos bien definidos. La separación entre géneros acústicos (Classical/Acoustic) y eléctricos (Rock/Dance) es casi perfecta, existiendo solo una pequeña confusión en las fronteras difusas (subgéneros híbridos).

---

## 🚀 Extra: Aplicación MLOps

Como complemento al análisis, se ha incluido en la carpeta `/app` una pequeña demostración de **Productivización del Modelo**.

Se trata de un script en Streamlit (`spotify_recommender.py`) que carga el modelo entrenado y permite realizar inferencias en tiempo real, además de incluir un sistema de recomendación básico mediante KNN.

---

## ⚙️ Reproducibilidad

Para ejecutar el notebook en local:

1.  Clonar el repositorio.
2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Lanzar Jupyter:
    ```bash
    jupyter notebook Music_Genre_Classification.ipynb
    ```

---
*Proyecto realizado para la asignatura de Aprendizaje Computacional.*
