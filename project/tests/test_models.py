import os
import joblib
import pytest
from sklearn.metrics import accuracy_score

import sys
import os

# Adicione o caminho raiz do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.load_data import load_data

def test_model_saving():
    """
    Verifica se o modelo foi salvo corretamente no diretório correto.
    """
    model_path = 'src/models/train_saved/model_lr.pkl'
    assert os.path.exists(model_path), "O modelo não foi salvo corretamente!"

def test_model_performance():
    """
    Verifica se o desempenho do modelo atende às expectativas.
    """
    model_path = 'src/models/train_saved/model_lr.pkl'
    model = joblib.load(model_path)

    # Carregar dados de teste
    X_test = joblib.load('src/data/processed_data/X_test.pkl')
    y_test = joblib.load('src/data/processed_data/y_test.pkl')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Defina um limite mínimo de acurácia esperado
    assert accuracy >= 0.75, f"A acurácia do modelo é muito baixa! ({accuracy:.2f})"