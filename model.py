import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


class RiskModel:

  def __init__(self, model_path='models/risk_model.joblib'):
    self.model_path = model_path
    self.model = None
    self.feature_names = []

    # Cargar modelo preentrenado si ya existe
    if os.path.exists(self.model_path):
      self.load_model()

  def train(self, df: pd.DataFrame, target_col='es_asegurable'):
    """Entrena el modelo de clasificación con los datos preprocesados de data_layer.py

    y exporta el archivo .joblib.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    self.feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inicializar clasificador (puedes cambiar a LightGBM o XGBoost)
    self.model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    self.model.fit(X_train, y_train)

    # Evaluación rápida de métricas
    y_pred = self.model.predict(X_test)
    y_proba = self.model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f'Modelo entrenado exitosamente. Accuracy: {acc:.4f} | AUC: {auc:.4f}')

    # Guardar modelo en disco
    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    joblib.dump(
        {'model': self.model, 'features': self.feature_names}, self.model_path
    )
    print(f'Modelo exportado en: {self.model_path}')

  def load_model(self):
    """Carga el modelo y la lista de features desde el archivo guardado."""
    data = joblib.load(self.model_path)
    self.model = data['model']
    self.feature_names = data['features']
    print(f'Modelo cargado correctamente desde {self.model_path}')

  def predict_risk(self, input_data: dict) -> dict:
    """Recibe los datos del cliente (desde routes.py) y devuelve la evaluación de riesgo.

    - decision: "Asegurable" / "No asegurable"
    - porcentaje_riesgo: float (0 - 100%)
    """
    if self.model is None:
      raise ValueError(
          'El modelo no ha sido cargado ni entrenado previamente.'
      )

    # Convertir diccionario a DataFrame alineado con las features de entrenamiento
    input_df = pd.DataFrame([input_data])

    # Rellenar variables faltantes con 0 para evitar errores de shape
    for col in self.feature_names:
      if col not in input_df.columns:
        input_df[col] = 0

    input_df = input_df[self.feature_names]

    # Probabilidad de pertenecer a la clase 0 (Clase 0 = Inasegurable / Alto Riesgo)
    prob_clase_asegurable = self.model.predict_proba(input_df)[0][1]
    porcentaje_riesgo = round((1 - prob_clase_asegurable) * 100, 2)

    prediccion = self.model.predict(input_df)[0]
    decision = 'Asegurable' if prediccion == 1 else 'No asegurable'

    return {
        'decision': decision,
        'porcentaje_riesgo': porcentaje_riesgo,
        'prob_asegurable': round(prob_clase_asegurable, 4),
    }

  @staticmethod
  def calcular_prima_estimada(
      suma_asegurada: float,
      porcentaje_riesgo: float,
      edad: int,
      fuma: bool = False,
  ) -> float:
    """Calcula la prima estimada según la regla actuarial simplificada para seguros de vida en Argentina.

    Fórmula base habitual: Prima = (Tasa Base * Factor Riesgo * Factor Edad *
    Factor Hábitos) * (Suma Asegurada / 1000)
    """
    tasa_base_por_mil = 1.5  # Tasa base por cada $1000 asegurados

    # Factor de riesgo derivado del modelo ML (entre 0.8 y 2.5)
    factor_riesgo = 0.8 + (porcentaje_riesgo / 100.0) * 1.7

    # Recargo actuarial por edad
    factor_edad = 1.0
    if edad > 60:
      factor_edad = 2.0
    elif edad > 45:
      factor_edad = 1.5
    elif edad > 30:
      factor_edad = 1.2

    # Recargo por tabaquismo
    factor_habitos = 1.35 if fuma else 1.0

    prima_anual = (
        suma_asegurada / 1000.0
    ) * (tasa_base_por_mil * factor_riesgo * factor_edad * factor_habitos)

    return round(prima_anual, 2)