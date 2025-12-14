================================================================================
RELATÓRIO TÉCNICO: CLASSIFICADOR BASEADO EM VIZINHANÇA COM PRIVACIDADE DIFERENCIAL
================================================================================

1. VISÃO GERAL
--------------------------------------------------------------------------------
Este projeto implementa um classificador supervisionado baseado em instâncias 
(semelhante ao k-NN, mas utilizando Raio Fixo) aplicado ao conjunto de dados 
"Adult Income" (Census Income). 

O objetivo é prever a IDADE ('age') de um indivíduo com base em seus outros 
atributos censitários, comparando o desempenho de uma abordagem tradicional 
versus uma abordagem com Privacidade Diferencial (Mecanismo de Laplace).

2. DETALHES DA IMPLEMENTAÇÃO
--------------------------------------------------------------------------------
Linguagem: Python 3.x
Bibliotecas: Numpy, Pandas, Matplotlib, Kagglehub.

A. PRÉ-PROCESSAMENTO:
   - Remoção de colunas irrelevantes: 'fnlwgt', 'education', 'capital-gain', etc.
   - Limpeza de dados: Remoção de linhas contendo dados faltantes ('?').
   - Codificação: Conversão de variáveis categóricas para numéricas utilizando 
     'pd.factorize' (Label Encoding), permitindo o cálculo de distâncias geométricas.
   - Divisão: Treino (70%) e Teste (30%) com embaralhamento aleatório.

B. ALGORITMO DE CLASSIFICAÇÃO (Raio Fixo):
   Ao contrário do k-NN padrão que busca os 'k' vizinhos, este algoritmo utiliza 
   uma abordagem baseada em Raio (Radius Neighbors):
   1. Para cada instância de teste, calcula a distância Euclidiana para todo o treino.
   2. Seleciona todos os vizinhos que estão dentro de um raio de distância (r <= 6).
   3. Realiza a votação com base nas classes (idades) desses vizinhos.

C. MECANISMO DE PRIVACIDADE (LAPLACE):
   A privacidade é garantida na etapa de agregação dos votos (Output Perturbation).
   
   - Sensibilidade: A sensibilidade global é considerada em relação ao histograma 
     de contagem de votos.
   - Composição do Orçamento: O orçamento epsilon é dividido pelo número de classes 
     únicas (L) presentes no treino (idades possíveis).
     -> Epsilon_classe = Epsilon / L
     -> Escala do Ruído (b) = 1 / Epsilon_classe = L / Epsilon
   
   Isso garante que a distribuição de frequência das idades dos vizinhos seja 
   ofuscada antes de decidir a classe vencedora.

3. PARÂMETROS UTILIZADOS
--------------------------------------------------------------------------------
- Raio de vizinhança (r): 6.0
- Orçamentos de Privacidade (Epsilon): [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  * Epsilon 0.5: Muito Ruído (Alta Privacidade / Baixa Utilidade).
  * Epsilon 3.0: Pouco Ruído (Baixa Privacidade / Alta Utilidade).

4. RESULTADOS E MÉTRICAS
--------------------------------------------------------------------------------
O sistema gera:
1. Arquivos .csv com as predições para cada nível de privacidade.
2. Cálculo de Acurácia (Taxa de acerto exato da idade).
3. Gráfico de linha comparando a recuperação da acurácia conforme o Epsilon aumenta.

Nota: Como a variável alvo é 'age' (muitas classes possíveis, ex: 18 a 90 anos), 
a acurácia absoluta tende a ser naturalmente baixa. O foco da análise é a 
diferença relativa (gap) entre o modelo tradicional e o privado.

================================================================================