Com base no código que você implementou e na teoria de Privacidade Diferencial, aqui estão as respostas detalhadas para os itens solicitados para o seu relatório ou trabalho.

---

### 1. Pseudo-código do kNN (Com e Sem Privacidade)

Aqui descrevemos o fluxo do algoritmo. A diferença crucial ocorre na etapa de "Votação".

**Entrada:** Conjunto de Treino $(X_{train}, Y_{train})$, Instância de Teste $x_{query}$, Hiperparâmetro $k$, Orçamento de privacidade $\epsilon$.
**Saída:** Classe prevista $\hat{y}$.

1.  **Carregar dados:** Armazenar $X_{train}$ e $Y_{train}$ na memória.
2.  **Para cada** instância de teste $x_{query}$:
3.  **Calcular Distâncias:**
    *   Calcular a distância Euclidiana entre $x_{query}$ e todos os pontos em $X_{train}$.
    *   $d(x_{query}, x_i) = \sqrt{\sum (x_{query} - x_i)^2}$
4.  **Identificar Vizinhos:**
    *   Ordenar as distâncias em ordem crescente.
    *   Selecionar os índices das $k$ menores distâncias.
    *   Recuperar os rótulos (classes) $Y$ desses $k$ vizinhos.
5.  **Contabilizar Votos (Contagem Real):**
    *   Para cada classe única $c$, calcular $count(c)$: quantos vizinhos pertencem à classe $c$.
6.  **Se kNN Tradicional:**
    *   Retornar a classe com o maior valor de $count(c)$.
7.  **Se kNN com Privacidade (Mecanismo de Laplace):**
    *   Definir sensibilidade $\Delta f = 1$.
    *   Calcular o parâmetro de escala $b = \Delta f / \epsilon$.
    *   Para cada classe $c$:
        *   Gerar ruído $ruido \sim Laplace(0, b)$.
        *   Calcula $voto\_ruidoso(c) = count(c) + ruido$.
    *   Retornar a classe com o maior valor de $voto\_ruidoso(c)$.

---

### 2. Como codificou atributos não numéricos

No código apresentado, a conversão é feita explicitamente na linha:
`self.X_train = np.array(X, dtype=np.float32)`

No entanto, para que essa linha funcione sem erro e para que a distância Euclidiana faça sentido, **o pré-processamento deve ter ocorrido antes de chamar a classe**. Seus dados brutos (strings, categorias) devem ter sido transformados em números.

**Resposta sugerida:**
> "Como o algoritmo utiliza a distância Euclidiana, que é uma medida geométrica, todos os atributos categóricos foram transformados em representações numéricas durante a etapa de pré-processamento (antes da execução do kNN). As técnicas utilizadas geralmente são **One-Hot Encoding** (para variáveis nominais sem ordem) ou **Label Encoding** (para variáveis ordinais ou binárias), garantindo que a matriz de entrada $X$ seja estritamente numérica (`float32`) para permitir operações algébricas."

---

### 3. Como aplicou o mecanismo de Laplace

O mecanismo foi aplicado na etapa de agregação (votação), conhecido como **Perturbação do Resultado Intermediário**.

**Resposta sugerida:**
> "O mecanismo de Laplace foi aplicado sobre a contagem de votos das classes vizinhas (histograma de votos). Em vez de simplesmente escolher a classe majoritária entre os $k$ vizinhos, adicionou-se um ruído aleatório extraído de uma distribuição de Laplace, centrado em 0 com escala $b = 1/\epsilon$, a cada contagem de classe. A classe predita foi aquela que obteve a maior contagem após a adição do ruído. Isso mascara a influência exata de qualquer vizinho individual na decisão final."

---

### 4. Qual valor de orçamento utilizado e o porquê

Você utilizou os valores definidos na instrução: $\epsilon \in [0.5, 1, 5, 10]$.

**Resposta sugerida:**
> "O orçamento de privacidade ($\epsilon$) controla o *trade-off* entre privacidade e utilidade.
> *   **Para cada predição**, o orçamento total $\epsilon$ foi utilizado para calibrar o ruído da votação.
> *   **Valores Baixos ($\epsilon=0.5, 1$):** Indicam alta privacidade. O ruído adicionado é grande, o que protege a identidade dos vizinhos, mas pode alterar a classe vencedora incorretamente, diminuindo a acurácia.
> *   **Valores Altos ($\epsilon=5, 10$):** Indicam baixa privacidade. O ruído é pequeno, mantendo o resultado muito próximo do kNN original (alta acurácia), mas oferecendo menor proteção aos dados."

---

### 5. Qual a sensibilidade global ($\Delta f$) e o porquê?

**Valor:** $\Delta f = 1$.

**Resposta sugerida:**
> "A sensibilidade global utilizada foi **1**. Isso ocorre porque a função de consulta é uma **contagem de votos**. Ao modificar, adicionar ou remover um único registro da base de dados de treino, o conjunto de $k$ vizinhos mais próximos pode ser alterado. No pior caso, um vizinho da classe A é substituído por um vizinho da classe B. Isso altera a contagem da classe A em -1 ou da classe B em +1. Portanto, a magnitude máxima da mudança na contagem de qualquer classe dada a alteração de um único indivíduo é 1."

---

### 6. Gráfico de Acurácia por valor de $\epsilon$

Para gerar este gráfico, você precisa calcular a acurácia (porcentagem de acertos) comparando o arquivo `y_pred` gerado com o `y_true`.

Como não tenho os seus arquivos de resultado, criei o código Python abaixo que **calcula a acurácia automaticamente** e plota o gráfico. Você só precisa ter os arquivos CSV gerados pelo seu algoritmo na mesma pasta.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os

# Definição dos epsilons usados
epsilons = [0.5, 1, 5, 10]

# Lista para armazenar as acurácias
acuracias_privadas = []

# 1. Calcular Acurácia do kNN Tradicional (Baseline)
# Certifique-se que o arquivo existe
df_trad = pd.read_csv('resultado_knn_tradicional.csv')
acc_tradicional = accuracy_score(df_trad['y_true'], df_trad['y_pred'])
print(f"Acurácia Tradicional: {acc_tradicional:.4f}")

# 2. Calcular Acurácias dos Privados
for eps in epsilons:
    filename = f'resultado_knn_laplace_eps_{eps}.csv'
    if os.path.exists(filename):
        df_priv = pd.read_csv(filename)
        acc = accuracy_score(df_priv['y_true'], df_priv['y_pred'])
        acuracias_privadas.append(acc)
        print(f"Acurácia (eps={eps}): {acc:.4f}")
    else:
        print(f"Arquivo {filename} não encontrado!")
        acuracias_privadas.append(0) # Valor dummy se faltar arquivo

# 3. Gerar o Gráfico
plt.figure(figsize=(10, 6))

# Linha da privacidade diferencial
plt.plot(epsilons, acuracias_privadas, marker='o', linestyle='-', color='b', label='kNN com Laplace (Privado)')

# Linha de referência (kNN sem privacidade)
# Plotamos uma linha horizontal constante
plt.axhline(y=acc_tradicional, color='r', linestyle='--', label=f'kNN Tradicional (Acc: {acc_tradicional:.2f})')

# Estilização
plt.title('Impacto da Privacidade Diferencial na Acurácia do kNN')
plt.xlabel('Orçamento de Privacidade (epsilon)')
plt.ylabel('Acurácia')
plt.xticks(epsilons) # Garante que mostre apenas os epsilons do teste no eixo X
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Salvar ou mostrar
plt.savefig('grafico_acuracia_epsilon.png')
plt.show()
```

**O que esperar do gráfico:**
*   A linha azul (Privado) deve começar mais baixa em $\epsilon=0.5$ e subir conforme o $\epsilon$ aumenta.
*   Em $\epsilon=10$, a linha azul deve estar muito próxima ou igual à linha vermelha tracejada (Tradicional), pois o ruído é muito baixo.