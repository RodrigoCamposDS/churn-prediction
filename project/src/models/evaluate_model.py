

from sklearn.metrics import confusion_matrix

def calcular_prejuizo(y_true, y_pred, custo_fp=10, custo_fn=100):
    """
    Calcula o prejuízo total com base na matriz de confusão.

    Parâmetros:
    - y_true: array-like, Valores reais.
    - y_pred: array-like, Valores previstos.
    - custo_fp: float, Custo unitário de Falsos Positivos.
    - custo_fn: float, Custo unitário de Falsos Negativos.

    Retorna:
    - prejuizo_total: float, Prejuízo total calculado.
    """
    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Cálculo do prejuízo
    prejuizo_total = (fp * custo_fp) + (fn * custo_fn)

    # Exibir resultados
    print(f"Falsos Positivos (FP): {fp} | Custo Unitário: R$ {custo_fp}")
    print(f"Falsos Negativos (FN): {fn} | Custo Unitário: R$ {custo_fn}")
    print(f"Prejuízo Total: R$ {prejuizo_total:.2f}")

    return prejuizo_total





import numpy as np

def calcular_variacao_relativa(test_scores, threshold=0.05):
    """
    Função para calcular a variação relativa (desvio padrão / média) das métricas de teste 
    e verificar se todas as variações estão dentro de um limite aceitável.

    Parâmetros:
    -----------
    test_scores : array-like
        As pontuações de teste para cada tamanho de treino (geralmente, saídas de `learning_curve`).
    threshold : float, default 0.05
        O limite aceitável para a variação relativa (default 5%).

    Retorna:
    --------
    variacao_relativa : array-like
        A variação relativa para cada ponto de corte de treino.
    tamanhos_altos : list
        Lista de tamanhos de treino onde a variação relativa excede o limite.
    """
    # Calcular a média e o desvio padrão das métricas de teste
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Calcular a variação relativa (desvio padrão / média)
    variacao_relativa = test_std / test_mean

    # Verificar se todas as variações estão dentro do limite
    tamanhos_altos = [i + 1 for i, var_rel in enumerate(variacao_relativa) if var_rel >= threshold]

    # Exibir os resultados
    print("Variação relativa (desvio padrão / média) por tamanho de treino:")
    for i, var_rel in enumerate(variacao_relativa):
        print(f"Tamanho {i + 1}: {var_rel:.4f}")
    
    if not tamanhos_altos:
        print(f"\nA variação está dentro dos limites aceitáveis (< {threshold*100}%).")
    else:
        print(f"\nA variação é alta em alguns tamanhos de treino (≥ {threshold*100}%).")
        print(f"Tamanhos com alta variação relativa: {tamanhos_altos}")

    return


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



import numpy as np
from sklearn.metrics import log_loss

def log_loss_final(model, X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Calcula a Log-Loss geral (somatório), Log-Loss para perdas/ganhos e o gradiente do modelo.

    Parâmetros:
        - model: Modelo treinado (ex: LogisticRegression)
        - X_train: Features de treino
        - y_train: Target de treino
        - X_test: Features de teste
        - y_test: Target de teste
        - threshold: Threshold ótimo para separar classes (padrão 0.5)

    Retorna:
        - log_loss_treino: Log-Loss para o conjunto de treino
        - log_loss_teste: Log-Loss para o conjunto de teste
        - log_loss_geral: Log-Loss média ponderada (treino + teste)
        - loss_log_loss: Log-Loss para classe 0 (perdas)
        - gain_log_loss: Log-Loss para classe 1 (ganhos)
        - gradient_norm: Norma do gradiente do modelo
    """
    # Probabilidades previstas para treino e teste
    probabilities_train = model.predict_proba(X_train)[:, 1]
    probabilities_test = model.predict_proba(X_test)[:, 1]
    
    # Log-Loss geral (treino e teste)
    log_loss_treino = log_loss(y_train, probabilities_train)
    log_loss_teste = log_loss(y_test, probabilities_test)

    # Log-Loss Geral (ponderada pelo tamanho dos conjuntos de treino e teste)
    total_instances = len(y_train) + len(y_test)
    log_loss_geral = (
        (len(y_train) * log_loss_treino + len(y_test) * log_loss_teste) / total_instances
    )

    # Aplicando o threshold ótimo
    y_pred_thresholded = (probabilities_test >= threshold).astype(int)
    
    # Índices de perda (classe 0) e ganho (classe 1)
    loss_indices = (y_test == 0)
    gain_indices = (y_test == 1)
    
    # Log-Loss para classe 0 (perdas)
    loss_log_loss = -np.mean(np.log(1 - probabilities_test[loss_indices])) if np.any(loss_indices) else 0
    
    # Log-Loss para classe 1 (ganhos)
    gain_log_loss = -np.mean(np.log(probabilities_test[gain_indices])) if np.any(gain_indices) else 0
    
    # Norma do gradiente do modelo
    gradient_norm = np.linalg.norm(model.coef_)
    
    # Exibição dos resultados
    print(f"Log Loss - Treino (Geral): {log_loss_treino:.4f}")
    print(f"Log Loss - Teste (Geral): {log_loss_teste:.4f}")
    print(f"Log Loss - Geral (Ponderada): {log_loss_geral:.4f}")
    print(f"Log Loss de Perda (classe 0) com threshold ótimo: {loss_log_loss:.4f}")
    print(f"Log Loss de Ganho (classe 1) com threshold ótimo: {gain_log_loss:.4f}")
    print(f"Norma do Gradiente: {gradient_norm:.4f}")
    
    # Retorna os valores calculados
    return gradient_norm, log_loss_geral
