import os
import pandas as pd
import joblib

# Caminhos dos arquivos relevantes
base_path = "/Users/rodrigocampos/Documents/data/case/Churned/project/src/data/processed_data"

arquivos = [
    "X_test.pkl", "X_train.pkl", "y_pred_lr.pkl", "y_pred_rf.pkl",
    "y_pred_th_otimo_lr.pkl", "y_pred_th_otimo_rf.pkl",
    "y_prob_th_lr.pkl", "y_prob_th_rf.pkl", "y_test.pkl", "y_train.pkl"
]

# Lista para armazenar os dados consolidados
dados_consolidados = []

# Loop para carregar os arquivos e adicionar ao DataFrame
for arquivo in arquivos:
    file_path = os.path.join(base_path, arquivo)
    try:
        # Carregar o arquivo com joblib
        data = joblib.load(file_path)
        
        # Verificar se o arquivo é uma matriz (array) e transformar em DataFrame
        if isinstance(data, (pd.DataFrame, pd.Series)):
            dados_consolidados.append(data)
        else:
            # Transformar arrays do NumPy em DataFrame
            dados_consolidados.append(pd.DataFrame(data))
        
        print(f"{arquivo}: Arquivo carregado e adicionado ao DataFrame!")
    except Exception as e:
        print(f"{arquivo}: Erro ao carregar arquivo - {e}")

# Concatenar todos os DataFrames em um único DataFrame
df_consolidado = pd.concat(dados_consolidados, axis=0, ignore_index=True)

# Salvar o DataFrame consolidado em um arquivo .pkl
output_path = os.path.join(base_path, "metrics_churn.pkl")
df_consolidado.to_pickle(output_path)

print(f"Arquivo consolidado salvo em: {output_path}")