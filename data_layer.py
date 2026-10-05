import pandas as pd

class data_layer:
    def __init__(self):
        self.data = []
    # 1. Cargar dataset
    df = pd.read_csv('csv_files/train.csv')
    # 2. Definir límite de nulos y eliminar columnas vacías
    umbral_nulos = 0.40  # 40%
    columnas_a_retener = df.columns[df.isnull().mean() < umbral_nulos]
    df_clean = df[columnas_a_retener].copy()

    # 3. Eliminar ID
    if 'Id' in df_clean.columns:
        df_clean.drop(columns=['Id'], inplace=True)

    # 4. Crear la variable objetivo binaria (Asegurable: 1, No asegurable: 0)
    # En Prudential, Response 6, 7 y 8 representan bajo riesgo
    df_clean['es_asegurable'] = (df_clean['Response'] >= 6).astype(int)
    df_clean.drop(columns=['Response'], inplace=True)

    # 5. Imputar valores faltantes restantes
    # Para variables numéricas usamos la mediana
    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

    print(f"Dataset reducido de {df.shape[1]} a {df_clean.shape[1]} columnas.")

    def add_data(self, item):
        self.data.append(item)

    def get_data(self):
        return self.data

    def clear_data(self):
        self.data = []