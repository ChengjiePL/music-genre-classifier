# 🎵 Music Genre Classification Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Un enfocament de Data Science rigorós per classificar gèneres musicals utilitzant característiques d'àudio de Spotify, des de l'Exploratory Data Analysis (EDA) fins al desplegament d'un model XGBoost optimitzat.**

---

## 📖 Descripció del Projecte

Aquest projecte desenvolupa un model de *Machine Learning* capaç de classificar cançons en quatre gèneres musicals distintius (**Acoustic, Classical, Dance, Hard-Rock**) basant-se exclusivament en les seves propietats acústiques (`energy`, `valence`, `tempo`, etc.).

L'objectiu no és només obtenir una alta precisió, sinó demostrar un **flux de treball científic complet**: des de la neteja de dades i l'enginyeria de característiques fins a l'avaluació probabilística avançada i la interpretació de models "Black Box".

### 🎯 Objectius Principals
1.  **Entendre les dades:** Analitzar com es diferencien els gèneres físicament mitjançant tècniques estadístiques i visuals (PCA, Correlacions).
2.  **Construir un classificador robust:** Superar el 85% d'accuracy minimitzant el *data leakage*.
3.  **Optimització científica:** Utilitzar tècniques avançades com *GridSearchCV* i *Cross-Validation* per garantir l'estabilitat.

---

## 🛠️ Tecnologies i Llibreries

*   **Llenguatge:** Python
*   **Manipulació de Dades:** Pandas, NumPy
*   **Visualització:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (PCA, Scaling, Metrics, RandomForest), XGBoost
*   **Validació:** K-Fold Cross Validation, ROC Curves

---

## 📊 Metodologia

El projecte segueix una estructura seqüencial rigorosa:

### 1. Preprocessament i Neteja 🧹
*   Reducció del dataset original (114k cançons) a un subconjunt equilibrat de 4.000 mostres per garantir qualitat sobre quantitat.
*   Eliminació de duplicats per `track_id` i combinacions `Nom + Artista`.
*   Neteja de metadades irrellevants per forçar l'aprenentatge basat en àudio.

### 2. Feature Engineering 🧪
Creació de variables sintètiques per capturar relacions no lineals:
*   `Intensity`: Combinació de *Loudness* i *Energy*.
*   `Dance_Tempo`: Relació entre ritme i ballabilitat.
*   `Chill_Factor`: Diferencial entre positivitat (*Valence*) i energia.

### 3. Exploratory Data Analysis (EDA) 📈
*   **Mapes de calor:** Detecció de multicolinealitat (ex: *Energy* vs *Loudness*).
*   **Boxplots:** Identificació de "signatures" de gènere (ex: la nul·la energia del *Classical* vs la saturació del *Hard-Rock*).

### 4. Modelatge i Optimització 🤖
S'han avaluat múltiples models, culminant en un **XGBoost Classifier**:
*   **Baseline (Random Forest):** 87.55% Accuracy.
*   **XGBoost (Tuned):** Optimització d'hiperparàmetres (GridSearchCV amb 72 candidats).
*   **Resultat Final:** **89.38% Accuracy** en Test.

---

## 🏆 Resultats Clau

El model final (XGBoost) ha demostrat una robustesa excepcional:

| Mètrica | Valor | Interpretació |
| :--- | :--- | :--- |
| **Accuracy** | **89.38%** | El model encerta gairebé 9 de cada 10 cançons. |
| **AUC (Mitjana)** | **0.98** | Capacitat quasi perfecta de rànquing probabilístic. |
| **Cross-Validation** | **88.69% (±1.9%)** | El model és estable i no depèn del split de dades. |

### Visualització de Rendiment

<p align="center">
  <!-- Pots substituir aquestes rutes per les imatges reals si les puges al repo -->
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="45%">
  <img src="assets/roc_curve.png" alt="ROC Curve" width="45%">
</p>

*   **Classical & Dance:** Gairebé perfectes (F1 > 0.90).
*   **Hard-Rock & Acoustic:** Petites confusions acceptables degut a solapaments espectrals visualitzats al PCA.

---

## 🚀 Com executar el projecte

1.  **Clonar el repositori:**
    ```bash
    git clone https://github.com/ChengjiePL/music-genre-classification.git
    cd music-genre-classification
    ```

2.  **Instal·lar dependències:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Executar el Notebook:**
    Obre `music_classification.ipynb` a Jupyter Lab o VS Code i executa les cel·les seqüencialment.

---

## 🧠 Conclusions i Aplicabilitat Real

Aquest projecte demostra que, tot i la complexitat de la música, les característiques d'àudio contenen patrons matemàtics clars que un model de *Gradient Boosting* pot desxifrar. 

**Aplicacions pràctiques:**
*   🎧 **Sistemes de Recomanació:** Suggerir cançons similars basant-se en l'àudio, no en l'artista.
*   📂 **Organització Automàtica:** Classificació de biblioteques musicals personals.
*   📻 **Generació de Playlists:** Creació de llistes per "estat d'ànim" (ex: filtrar per *Chill_Factor* alt).

---

## 👤 Autor

**ChengjiePL**  
*Data Science Student & Developer*

---

> *Aquest projecte ha estat realitzat amb finalitats acadèmiques, buscant l'excel·lència en la metodologia de Data Science.*

