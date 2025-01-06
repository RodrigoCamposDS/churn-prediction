from relatorio import gerar_relatorio

# Exemplo de dados (esses dados podem vir das suas matrizes de confusão)
resultados = [
    {"Modelo": "Reg. Logística", "Threshold": 0.5, "FP": 875, "Custo_FP": 4375, "FN": 1203, "Custo_FN": 30075, "Custo_Total": 34450},
    {"Modelo": "Reg. Logística", "Threshold": 0.347, "FP": 1177, "Custo_FP": 5885, "FN": 621, "Custo_FN": 15525, "Custo_Total": 21410},
    {"Modelo": "Random Forest", "Threshold": 0.5, "FP": 888, "Custo_FP": 4440, "FN": 1903, "Custo_FN": 47575, "Custo_Total": 52015},
    {"Modelo": "Random Forest", "Threshold": 0.367, "FP": 1771, "Custo_FP": 8855, "FN": 1272, "Custo_FN": 31800, "Custo_Total": 40655}
]

# Gerando o relatório
gerar_relatorio(resultados)