import pandas as pd

def load_data(file_path):
    """
    Função para carregar o dataset.
    """
    return pd.read_csv(file_path)