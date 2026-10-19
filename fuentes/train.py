#entrenamiento y evaluacion de modelos
import os
import time
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#cargar los datos limpios
ruta_datos = "../datos/datos_limp/bitext_limpio.csv"
datos = pd.read_csv(ruta_datos)

X = datos["instruction"]
y = datos["category"]

#separar datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#configurar la validacion cruzada
validacion = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

#configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Clasificacion_Mensajes_Atencion")

#funcion para entrenar, evaluar y registrar un modelo
def evaluar_modelo(nombre, pipeline, parametros):
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring="f1_macro",
        cv=validacion,
        n_jobs=-1
    )

    inicio = time.time()
    busqueda.fit(X_train, y_train)
    tiempo = time.time() - inicio

    predicciones = busqueda.predict(X_test)

    #calcular las metricas
    metricas = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision_macro": precision_score(
            y_test, predicciones, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predicciones, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predicciones, average="macro", zero_division=0
        ),
        "tiempo_busqueda_segundos": tiempo
    }

    #registrar resultados en MLflow
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metrics(metricas)
        mlflow.log_metric(
            "f1_macro_validacion", busqueda.best_score_
        )
        mlflow.sklearn.log_model(
            sk_model=busqueda.best_estimator_,
            name="modelo_" + nombre.lower().replace(" ", "_")
        )

    print("\nModelo:", nombre)
    print("Mejores parametros:", busqueda.best_params_)
    print("Metricas:", metricas)

#modelo 1: Regresion Logistica
pipeline_logistica = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("modelo", LogisticRegression(max_iter=1000))
])

parametros_logistica = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "modelo__C": [0.5, 1.0, 2.0]
}

evaluar_modelo(
    "Regresion_Logistica",
    pipeline_logistica,
    parametros_logistica
)

#modelo 2: Naive Bayes
pipeline_nb = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("modelo", MultinomialNB())
])

parametros_nb = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "modelo__alpha": [0.5, 1.0, 1.5]
}

evaluar_modelo(
    "Naive_Bayes",
    pipeline_nb,
    parametros_nb
)
