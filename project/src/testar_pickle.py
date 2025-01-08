import joblib

arquivos = [
    "X_test.pkl", "X_train.pkl", "y_pred_lr.pkl", "y_pred_rf.pkl",
    "y_pred_th_otimo_lr.pkl", "y_pred_th_otimo_rf.pkl",
    "y_prob_th_lr.pkl", "y_prob_th_rf.pkl", "y_test.pkl", "y_train.pkl"
]
base_path = "/Users/rodrigocampos/Documents/data/case/Churned/project/src/data/processed_data"

for arquivo in arquivos:
    file_path = f"{base_path}/{arquivo}"
    try:
        # Use joblib em vez de pickle
        data = joblib.load(file_path)
        print(f"{arquivo}: Arquivo carregado com sucesso usando joblib!")
    except Exception as e:
        print(f"{arquivo}: Erro ao carregar arquivo com joblib - {e}")