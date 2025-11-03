Trabalho prático 1 - SAVI
==============
Miguel Riem Oliveira <mriem@ua.pt>
2025-2026

# Trabalho Prático: SAVICP 

**Registo de Nuvens de Pointos de Scans RGB-D com ICP Personalizado**


## Introdução

O registo de scans 3D é um problema fundamental em visão computacional e robótica, essencial para a reconstrução 3D, fusão de sensores, SLAM (Simultaneous Localization and Mapping) e muitas outras aplicações. O objetivo é alinhar múltiplas nuvens de pontos ou scans 3D de modo a que representem a mesma cena de forma consistente.

O algoritmo **Iterative Closest Point (ICP)** é uma das técnicas mais populares para este fim. O ICP funciona de forma iterativa, executando os seguintes passos:
1.  **Associação de Correspondências:** Para cada ponto na nuvem de pontos *fonte*, encontra-se o ponto mais próximo na nuvem de pontos *alvo*;
2.  **Estimação da Transformação:** Calcula-se a transformação rígida (rotação e translação) que melhor alinha as correspondências encontradas, minimizando uma função de erro (geralmente a soma dos quadrados das distâncias);
3.  **Aplicação da Transformação:** A nuvem de pontos *fonte* é transformada com a transformação calculada;
4.  **Iteração:** Os passos 1 a 3 são repetidos até que a transformação converja ou um número máximo de iterações seja atingido.

Neste trabalho prático, os alunos irão explorar o ICP em diferentes níveis de abstração, desde a utilização de implementações existentes até à construção de um otimizador personalizado, integrando as bibliotecas **OpenCV**, **Open3D** e **SciPy**.

## Configuração e Pré-requisitos

Certifique-se de que tem as seguintes bibliotecas instaladas no seu ambiente Python:
*   `open3d`
*   `opencv-python`
*   `scipy`
*   `numpy`
*   `matplotlib` (para visualização, se necessário)

**Dados:** Serão fornecidos conjuntos de imagens RGB e de profundidade (RGB-D), juntamente com os parâmetros intrínsecos da câmara, tipicamente do formato PNG (RGB) e PNG de 16 bits (profundidade).

### Processamento de Dados Comum a Todas as Tarefas

Antes de iniciar as tarefas específicas, terá de implementar as seguintes etapas para cada par de imagens RGB-D:

1.  **Carregamento de Imagens e Filtragem de Profundidade (OpenCV):**
2.  **Criação de Nuvens de Pontos (Open3D a partir de dados OpenCV):**
    *   Converta as imagens RGB e de profundidade (agora como arrays NumPy do OpenCV) em objetos `open3d.geometry.Image`.
    *   Crie um objeto `open3d.geometry.RGBDImage` a partir destas imagens.
    *   Utilize `open3d.geometry.PointCloud.create_from_rgbd_image` com os parâmetros intrínsecos da câmara para gerar objetos `open3d.geometry.PointCloud`.
3.  **Pré-processamento de Nuvens de Pontos (Open3D):**
    *   **Downsampling (Subamostragem):** Aplique `voxel_downsample` para reduzir a densidade das nuvens de pontos, acelerando o processamento.
    *   **Estimação de Normais:** Calcule os vetores normais para cada ponto da nuvem de pontos usando `estimate_normals`. As normais são cruciais para a variante Point-to-Plane do ICP bem como para uma melhor visualização da point cloud.

## Tarefas

---

### Tarefa 1: Registo ICP com as Ferramentas Nativas do Open3D (Referência)

**Objetivo:** Familiarizar-se com o algoritmo ICP e entender o seu funcionamento básico usando a implementação otimizada do Open3D.

1.  **Implementação:** Utilize a função `open3d.pipelines.registration.registration_icp` para registar um par de nuvens de pontos (fonte e alvo).
2.  **Configuração:** Experimente diferentes parâmetros, como `max_correspondence_distance` e `estimation_method` (e.g., `PointToPoint` vs. `PointToPlane`).
3.  **Inicialização:** Para esta tarefa, pode usar uma transformação inicial de identidade ou uma transformação manual aproximada obtida com recurso ao cloud compare.
4.  **Visualização:** Utilize `open3d.visualization.draw_geometries` para visualizar as nuvens de pontos antes e depois do registo.

**Deliverable:** Código Python **main_ipc.py** que demonstra o registo ICP usando o Open3D. Inclua a visualização dos resultados.

---

### Tarefa 2: ICP Personalizado com Otimização Least-Squares e Inicialização Manual

**Objetivo:** Implementar o ciclo ICP e a parte de otimização de raiz, usando `scipy.optimize.least_squares` para resolver o problema de minimização em cada iteração.

1.  **Ciclo ICP Personalizado:**
    *   Crie uma função para o algoritmo ICP que recebe duas nuvens de pontos e uma transformação inicial.
    *   Dentro do ciclo iterativo:
        *   **Correspondências (Open3D):** Para cada ponto na nuvem de pontos *fonte* (transformada na iteração anterior), encontre o seu vizinho mais próximo na nuvem de pontos *alvo* usando um `open3d.geometry.KDTreeFlann` ou outra solução;
        *   **Função de Custo:** Defina uma função de custo que, dado um conjunto de parâmetros de transformação (e.g., rotação e translação como um vetor de 6 elementos), calcule a soma das distâncias ao quadrado entre os pontos correspondentes (e.g., Point-to-Plane).
            *   *Sugestão:* A função de erro deve retornar um array das distâncias individuais para cada ponto na nuvem de pointos *fonte*;
        *   **Otimização (SciPy):** Utilize `scipy.optimize.least_squares` para encontrar os parâmetros da transformação *incremental* que minimizam a função de erro para as correspondências atuais.
            *   **Nota:** `scipy.optimize.least_squares` é uma ferramenta poderosa para otimização não-linear. O seu trabalho será definir a função que calcula o *residual* para cada par de correspondência.
        *   **Aplicação da Transformação (Open3D):** Aplique a transformação estimada à nuvem de pontos *fonte* usando métodos do Open3D (`transform`).
    *   **Critério de Paragem:** Defina um critério de paragem (e.g., número máximo de iterações, ou um valor de erro menor que o limite definido como mínimo).
2.  **Inicialização Manual:**
    *   Para esta tarefa, utilize uma transformação inicial manual. Pode obtê-la visualmente (e.g., usando ferramentas como o CloudCompare para alinhar as nuvens de pontos e exportar a matriz de transformação) ou uma estimativa aproximada.
3.  **Visualização:** Mostre o progresso do ICP e o resultado final.

**Deliverable:** Código Python do seu ICP personalizado **main_custom_icp.py**, usando `scipy.optimize.least_squares` e uma inicialização manual. Inclua a visualização.

---

### Tarefa 3: Otimização da Esfera Englobante Mínima

**Objetivo:** Adicionar parâmetros de uma esfera à otimização anterior para encontrar a menor esfera possível que contenha todos os pontos de duas nuvens de pontos especificadas.

1.  **Definição dos Parâmetros da Esfera:**
    *   Acrescente à otimização os seguintes parâmetros para a esfera:
        *   Coordenadas do centro: `(xc, yc, zc)`
        *   Raio da esfera: `r`
2.  **Formulação do Problema de Otimização:**
    *   **Função Objetivo:** Minimizar o raio `r` da esfera.
    *   **Restrições:** Todos os pontos pertencentes às nuvens de pontos 1 e 2 deverão estar contidos dentro da esfera. Isso significa que, para cada ponto `p = (x, y, z)` de ambas as nuvens, a distância euclidiana do ponto `p` ao centro `(xc, yc, zc)` da esfera deve ser menor ou igual ao raio `r`.
        *   `sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2) <= r` para todos os pontos.

**Deliverable:** Código Python **main_minimum_enclosing_sphere.py** que implementa a otimização para determinar os parâmetros `(xc, yc, zc)` e `r` da esfera englobante mínima. Inclua uma breve análise dos resultados obtidos.
---

## Entrega

Para cada tarefa, deverá submeter:
*   O código Python (`.py`) claro, comentado e funcional.
*   A entrega é feita com um repositório chamado savi-2025-2026-trabalho1-grupoX, em que X é o número do grupo. 
* O `README.md` deve conter a explicação do trabalho e uma descrição de cada tarefa e da sua resolução, nomeadamente explicando:
    *   A abordagem utilizada.
    *   Os resultados obtidos (incluindo capturas de ecrã das visualizações).
    *   Qualquer desafio encontrado e como foi resolvido.
    *   Comparações e análises conforme solicitado em cada tarefa.

## Dicas e Sugestões

*   **Visualização é Chave:** Utilize a visualização do Open3D para depurar o seu código e entender o que está a acontecer em cada etapa do ICP.
*   **Parâmetros:** A escolha de parâmetros como `max_correspondence_distance` é crucial para o desempenho do ICP.
*   **OpenCV e Open3D:** Explore as documentações de ambas as bibliotecas para descobrir funções úteis para filtragem, transformações e representação de dados.
*   **ChatGPT e Gemini:** Peça ajuda ao chatgpt para lhe dar sugestões de código para resolver as tarefas. Se não compreender o código, peça ajuda ao Gemini para explicar o código gerado pelo chatgpt. Se não perceber a explicação, peça ao chatgpt para explicar a explicação do Gemini do código do chatgpt, etc, etc.

Bom trabalho!