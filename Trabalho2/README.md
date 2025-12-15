# Trabalho Prático 2 - SAVI
==============

Miguel Riem Oliveira <mriem@ua.pt>
2025-2026

# Trabalho Prático: SAVI-Mnist

**Classificação e Deteção de Dígitos Manuscritos com Redes Neuronais Convolucionais**

## Introdução

Enquanto o primeiro trabalho focou em métodos clássicos de visão computacional (geometria e registo), este segundo trabalho foca-se na aprendizagem profunda (**Deep Learning**). O objetivo é evoluir de um problema de classificação simples (MNIST clássico) para um cenário mais realista e complexo: a deteção e classificação de múltiplos objetos em imagens maiores.

Os alunos irão consolidar os conhecimentos adquiridos sobre **PyTorch**, arquiteturas **CNN** (Convolutional Neural Networks), métricas de avaliação e técnicas de deteção de objetos. O trabalho evolui incrementalmente desde a otimização de um classificador até à implementação de um detetor de objetos completo.

## Configuração e Pré-requisitos

Certifique-se de que tem as seguintes bibliotecas instaladas no seu ambiente Python (além das já utilizadas no TP1):
*   `torch` e `torchvision` (PyTorch)
*   `scikit-learn` (para cálculo de métricas avançadas)
*   `tqdm` (para barras de progresso)
*   `seaborn` (para visualização de matrizes de confusão)
*   `git` (para clonar o repositório de geração de dataset)

**Dados Base:** O dataset MNIST será descarregado automaticamente via `torchvision` para a Tarefa 1. Para as tarefas seguintes, será gerado um dataset sintético.

## Tarefas

---

### Tarefa 1: Classificador CNN Otimizado (MNIST Completo)

**Objetivo:** Melhorar o classificador base desenvolvido nas aulas, utilizando o dataset MNIST completo e arquiteturas mais robustas.

1.  **Código Base:** Parta do código desenvolvido nas aulas (`main.py`, `model.py`, `trainer.py`).
2.  **Dataset Completo:** Ao contrário das aulas (onde usámos uma percentagem reduzida), configure o `dataset.py` para utilizar a totalidade dos dados de treino (60.000 imagens) e teste (10.000 imagens).
3.  **Melhoria da Arquitetura:**
    *   Altere o `model.py` para criar uma nova classe (e.g., `ModelBetterCNN`).
    *   Experimente adicionar mais camadas convolucionais, camadas de *Dropout* para regularização, e *Batch Normalization*.
    *   O objetivo é maximizar a accuracy no conjunto de teste.
4.  **Avaliação Detalhada:**
    *   Implemente o cálculo e visualização da **Matriz de Confusão**.
    *   Calcule e apresente as métricas de **Precision**, **Recall** e **F1-Score** (por classe e a média global/macro). Pode utilizar o `sklearn.metrics`.

**Deliverable:** Código Python **main_classification.py** e módulos associados. O `README` deve conter a tabela de resultados e a imagem da matriz de confusão.

---

### Tarefa 2: Geração de Dataset de "Cenas" com Dígitos

**Objetivo:** Criar um dataset mais complexo onde os dígitos não estão centrados e podem existir múltiplos dígitos numa imagem (simulando um cenário de deteção de objetos).


1.  Inspirado na ferramenta [MNIST-ObjectDetection](https://github.com/hukkelas/MNIST-ObjectDetection) desenvolva código para gerar digitos em imagens maiores
2.  **Modificações:**: **MNIST-ObjectDetection** não tem algumas funcionalidades desejadas. Em primeiro ligar o link para download do dataset está incorreto. Agora é "https://ossci-datasets.s3.amazonaws.com/mnist/". Depois pretende-se adicionar uma forma de evitar que as imagens geradas tenham digitos sobrepostos. Pretende-se também que os digitos tenham escalas pouco variadas (e.g. digitos com tamanho de 22x22 a 36x36).
2.  **Geração:**
    *   Gere um dataset com dimensões compatíveis com o MNIST original (ex: 60k treino, 10k teste) ou uma dimensão representativa que o seu hardware suporte bem.
    *   As imagens geradas devem ter dimensões maiores (e.g., 100x100 ou 128x128) e conter dígitos espalhados.
3.  **Experimentação:** Gere pelo menos duas versões do dataset para análise:
    *   *Versão A:* Apenas 1 dígito por imagem (mas em posição aleatória).
    *   *Versão B:* Apenas 1 dígito por imagem (mas em posição aleatória e com diferenças de escala).
    *   *Versão C:* Múltiplos dígitos por imagem (e.g., entre 3 a 5 dígitos).
    *   *Versão D:* Múltiplos dígitos por imagem (e.g., entre 3 a 5 dígitos com diferenças de escala).
4.  **Análise e Visualização:**
    *   Crie um script para visualizar mosaicos de imagens geradas com as respetivas "bounding boxes" (ground truth).
    *   Apresente estatísticas: distribuição de classes nos novos datasets, histograma de número de dígitos por imagem, tamanho médio dos dígitos, etc.

**Deliverable:** Script de geração ou descrição do comando utilizado. Script **main_dataset_stats.py** que gera as visualizações e estatísticas.

---

### Tarefa 3: Deteção por Janela Deslizante (Sliding Window)

**Objetivo:** Utilizar o classificador treinado na Tarefa 1 para encontrar dígitos nas "cenas" da Tarefa 2, sem re-treinar a rede.

1.  **Abordagem:** Implemente uma técnica de *Sliding Window* (Janela Deslizante).
    *   Percorra a imagem de entrada (do dataset da Tarefa 2) com janelas de tamanho 28x28 (ou redimensionadas).
    *   Passe cada recorte (crop) pela rede treinada na Tarefa 1.
2.  **Thresholding:** Defina um limiar de confiança (baseado na saída *softmax*) para decidir se um recorte contém um dígito ou é fundo (background).
    *   *Nota:* Como a rede da Tarefa 1 nunca viu "fundo", ela tentará classificar tudo como um dígito. Terá de lidar com este problema (e.g., analisando a entropia da saída ou a magnitude dos logits).
3.  **Visualização:** Desenhe as caixas delimitadoras (bounding boxes) onde a rede detetou dígitos com alta confiança sobre a imagem original.
4.  **Avaliação Qualitativa:** Discuta no README a eficiência desta abordagem (tempo de execução) e os problemas encontrados (falsos positivos, precisão da localização).

**Deliverable:** Código Python **main_sliding_window.py**. Inclua exemplos de imagens com as deteções no README.

---

### Tarefa 4: Detetor e Classificador Integrado

**Objetivo:** Alterar a arquitetura ou a estratégia de treino para realizar a deteção e classificação de forma mais eficiente e precisa.

1.  **Nova Abordagem:** Desenvolva uma solução que supere as limitações da janela deslizante. Algumas sugestões:
    *   **Conversão para FCN:** Converta as camadas `Linear` (fully connected) da sua CNN em camadas Convolucionais (Fully Convolutional Network). Isso permite passar a imagem inteira de uma vez e obter um mapa de calor de ativações.
    *   **Regressão de Bounding Box:** Altere a saída da rede para prever também as coordenadas `(x, y, w, h)` além da classe (abordagem simplificada tipo YOLO/R-CNN).
    *   **Region Proposals (RPN)**: Implemente um mecanismo de "Propostas de Região". Pode ser uma sub-rede dedicada (Region Proposal Network) que aprende a sugerir onde existem objetos antes de classificar (abordagem Two-Stage similar à Faster R-CNN), ou utilizar algoritmos rápidos de segmentação para gerar candidatos.
    *   **Re-treino:** Utilize o dataset da Tarefa 2 para treinar esta nova rede, permitindo que ela aprenda a distinguir "fundo" de "dígito".
2.  **Implementação:** Crie o treino e a inferência para esta nova arquitetura.
3.  **Comparação:** Compare os resultados (visuais e, se possível, de métricas) com a abordagem da Tarefa 3. A nova abordagem é mais rápida? É mais precisa?

**Deliverable:** Código Python **main_improved_detection.py**. Relatório detalhando as alterações arquiteturais feitas.

---

## Entrega

Para cada tarefa, deverá submeter:
*   O código Python (`.py`) claro, comentado e funcional.
*   A entrega é feita com um repositório chamado `savi-2025-2026-trabalho2-grupoX`, em que X é o número do grupo. 
*   O `README.md` deve ser o relatório principal, contendo:
    *   **Metodologia:** Explicação das arquiteturas de rede escolhidas (desenhos/diagramas são valorizados).
    *   **Resultados T1:** Matrizes de confusão e tabela de métricas (F1, Precision, Recall).
    *   **Análise de Dados T2:** Estatísticas e exemplos do dataset gerado.
    *   **Deteção T3 vs T4:** Comparação visual e discussão sobre performance (tempo vs qualidade) entre a janela deslizante e a abordagem melhorada.
    *   **Dificuldades:** Descrição dos principais desafios e soluções encontradas.

## Dicas e Sugestões

*   **GPU:** O treino com o dataset completo e a geração de dados podem ser pesados. Use a GPU (CUDA) se disponível. Verifique sempre com `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
*   **Overfitting:** Se a accuracy de treino for muito superior à de teste na Tarefa 1, o modelo está em *overfitting*. Aumente o Dropout ou reduza a complexidade do modelo.
*   **Sliding Window:** Na Tarefa 3, o passo (stride) da janela deslizante afeta muito a velocidade. Um passo muito pequeno é lento, um passo muito grande pode "saltar" dígitos.
*   **Fundo (Background):** Na Tarefa 4, é crucial que a rede aprenda o que é "não-dígito". Considere ter uma 11ª classe para "Fundo" ou treinar a rede para ter baixa ativação em zonas vazias.
*   **ChatGPT e Gemini:** Pode e deve usar AI para ajudar explorar os problemas.

Bom trabalho!