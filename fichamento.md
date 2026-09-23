# Fichamento — Artigos relacionados ao desafio (G7 — Abdominal Trauma Detection)

---

## 1. The RSNA Abdominal Traumatic Injury CT (RATIC) Dataset

- **Autores / Ano:** Rudie JD, Lin HM, Ball RL, et al. (2024)
- **Link / DOI:** https://doi.org/10.1148/ryai.240101 — *Radiology: Artificial Intelligence*, 6(6):e240101
- **Objetivo do artigo:** Descrever a curadoria e a anotação do dataset RATIC, base do desafio RSNA 2023 Abdominal Trauma Detection.
- **Dataset usado:** 4.274 exames de TC abdominal/pélvica de pacientes adultos (≥18 anos), 6.481 séries de imagem, 23 instituições em 14 países e 6 continentes. Rótulos em `train_2024.csv` (presença/gravidade de lesão em fígado, rim, baço, intestino/mesentério e extravasamento ativo), `image_level_labels_2024.csv` (rótulos em nível de imagem para intestino/mesentério e extravasamento), `train_series_meta.csv` (fase de imagem, cobertura anatômica) e `train_demographics_2024.csv`. Segmentações pixel-a-pixel (NIfTI) disponíveis para 206 séries.
- **Métodos empregados:** Anotação por 43 radiologistas (média de 10,2 anos de experiência), rótulos de gravidade de órgão sólido (fígado/baço/rim) estabelecidos por votação majoritária entre 3 anotadores, divididos em baixo grau (AAST I–III) e alto grau (AAST IV–V). Segmentações geradas por nnU-Net treinado no TotalSegmentator, com correção manual. Rótulos de lesão intestinal/mesentérica e extravasamento definidos por consenso, em nível de imagem.
- **Principais métricas/resultados:** Não é um artigo de modelo — reporta apenas a composição do dataset: 1.316 dos 4.274 casos (30,8%) positivos para alguma lesão; 491 lesões hepáticas, 517 esplênicas, 370 renais, 133 intestinais, 336 com extravasamento ativo (Tabela 1 do artigo). Confirma o desbalanceamento observado na nossa EDA.
- **Limitações apontadas pelos próprios autores:** Rótulos de gravidade dependem de votação majoritária entre anotadores (há variabilidade interobservador documentada na literatura para o escore AAST); ausência de fase tardia de imagem (limita avaliação de lesão do sistema coletor renal); protocolos de aquisição heterogêneos entre instituições (bifásico, split bolus, fase portal única); anotação por plataforma web, sem acesso a monitores de alta resolução ou exames prévios do paciente.
- **O que aproveitamos para o nosso baseline:** É a referência primária — descreve exatamente os arquivos (`train_2024.csv` etc.) que estamos usando, confirma a divisão baixo/alto grau (AAST I–III vs IV–V) que orienta a formulação do problema, e documenta o desbalanceamento que já vimos na EDA e que se confirmou também na amostra ampliada de 500 pacientes do Marco 3 (ex: `bowel_injury` em apenas 2,6% dos pacientes).

---

## 2. Automated Spleen Injury Detection Using 3D Active Contours and Machine Learning

- **Autores / Ano:** Wang J, Wood A, Gao C, Najarian K, Gryak J. (2021)
- **Link / DOI:** https://doi.org/10.3390/e23040382 — *Entropy*, 23(4):382
- **Objetivo do artigo:** Propor um método automatizado para detectar lacerações esplênicas em TC usando segmentação por contornos ativos 3D seguida de classificação com features hand-crafted.
- **Dataset usado:** 99 exames de TC (Michigan Medicine + dataset público CIREN), 54 baços saudáveis e 45 lacerados, classificados pela escala AAST/AIS (graus I–V).
- **Métodos empregados:** Segmentação automática do baço (método próprio anterior, active contours). Extração de 4 famílias de features: histograma (entropia de Rényi, média, variância, assimetria, curtose), fractal (dimensão fractal por box-counting), Gabor (banco de 40 filtros, 5 escalas × 8 orientações) e forma (circularidade, excentricidade, orientação, diferença área/área convexa). Classificadores: Random Forest, Naive Bayes, SVM, k-NN ensemble, subspace discriminant ensemble — todos com validação cruzada de 5 folds. Comparação direta contra um pipeline de deep learning (ResNet-50 + LSTM).
- **Principais métricas/resultados:** Random Forest foi o melhor classificador clássico: AUC 0,91, F1 0,80 no teste. **Superou a abordagem de deep learning** (AUC 0,72), atribuído ao tamanho pequeno da amostra — achado direto que embasa a escolha metodológica do nosso próprio TP1 (métodos clássicos como baseline robusto em cenário de dados limitados).
- **Limitações apontadas pelos próprios autores:** Erros de classificação concentrados em lesões leves/moderadas (AIS 2–3), não nas graves; segmentações imperfeitas (6 de 99 casos descartados por erro de segmentação); dataset pequeno por grau de gravidade individual; extensão para multi-classe (por grau) e para outros tipos de lesão esplênica (hematomas, hemorragias) deixada para trabalhos futuros.
- **O que aproveitamos para o nosso baseline:** Repertório de descritores (Gabor, fractal, forma, histograma) aplicável ao baço/fígado/rim; validação empírica de que RF com features hand-crafted pode superar deep learning em dataset pequeno — argumento direto para a Introdução do nosso artigo; metodologia de partição por paciente e CV de 5 folds, igual à exigida no nosso protocolo. **Achado relacionado na nossa própria grade de experimentos (Marco 3):** ao contrário do que Wang et al. observaram para RF vs. deep learning, em *nosso* caso o SVM superou consistentemente Random Forest e XGBoost nos alvos mais raros (`bowel`, `kidney`) — sinal de que, entre os métodos clássicos, o reponderamento por margem do SVM (`class_weight="balanced"`) generaliza melhor que o reponderamento por split dos ensembles de árvore sob desbalanceamento severo (<5% de prevalência). Vale contrastar esse achado com o de Wang et al. na Discussão do artigo.

---

## 3. Diagnosis of Traumatic Liver Injury on CT Using Machine Learning Algorithms and Radiomics Features

- **Autores / Ano:** Alimiri Dehbaghi H, Khoshgard K, Sharini H, Khairabadi SJ. (2024)
- **Link / DOI:** https://doi.org/10.4103/jrms.jrms_847_23 — *Journal of Research in Medical Sciences*, 29:77
- **Objetivo do artigo:** Avaliar o desempenho de modelos de ML combinados a features radiômicas para diagnosticar e estadiar (leve/grave) lesão hepática traumática em TC.
- **Dataset usado:** 600 cortes axiais de TC do próprio dataset Kaggle RSNA 2023 Abdominal Trauma Detection (200 fígado saudável, 200 lesão leve, 200 lesão grave) — **mesmo dataset do nosso desafio**, usado em recorte 2D.
- **Métodos empregados:** Segmentação manual no 3D Slicer. Extração radiômica via toolbox do 3D Slicer: features de primeira ordem + textura (GLCM, GLDM, GLRLM, GLSZM, NGTDM), aplicadas sobre 8 decomposições wavelet (HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL). 30 modelos testados inicialmente, reduzidos a 3: LightGBM, Ridge Classifier, XGBoost. Split 75/25 treino/teste.
- **Principais métricas/resultados:** LightGBM foi o melhor: acurácia 94% (lesão leve) e 99% (lesão grave), AUC 93,8% e 99,0%, respectivamente. Tabela 1 do artigo lista as features radiômicas mais importantes por modelo — referência direta para nossa etapa de seleção de características.
- **Limitações apontadas pelos próprios autores:** Falta de acesso a informações clínicas do paciente (só imagem); segmentação manual (não automatizada); sugerem aumentar o número de sujeitos em trabalhos futuros.
- **O que aproveitamos para o nosso baseline:** É o artigo mais diretamente comparável — usa o mesmo dataset, radiômica IBSI-compatível (recomendada no enunciado do TP1) e um dos 3 classificadores clássicos que testamos (XGBoost). Serve de ponto de comparação numérico direto: nosso melhor resultado para `liver` no Marco 3 (0,456 de balanced accuracy, via intensidade + Random Forest) fica bem abaixo dos 94-99% reportados aqui — diferença esperada, já que o estudo usa segmentação manual do órgão e classes balanceadas por construção (200/200/200), enquanto nosso pipeline não segmenta o fígado e trabalha com o desbalanceamento real do dataset (90,8%/8,2%/1,0%). Essa diferença metodológica é um ponto importante a explicitar na Discussão.

---

## 4. Algoritmo diagnóstico baseado em TC para identificar lesões intestinais e/ou mesentéricas em trauma abdominal fechado

- **Autores / Ano:** Lansier A, Bourillon C, Cuénod CA, et al. (2023)
- **Link / DOI:** https://doi.org/10.1007/s00330-022-09200-9 — *European Radiology*, 33(3):1918–1927
- **Objetivo do artigo:** Desenvolver e validar um algoritmo diagnóstico baseado em sinais de TC (não radiômica/ML de imagem, mas achados semânticos) para lesão intestinal e/ou mesentérica em trauma abdominal fechado.
- **Dataset usado:** Coorte de treinamento com 79 pacientes (29 com lesão intestinal/mesentérica, 50 sem) e coorte de validação com 37 pacientes (13 com lesão, 24 sem). Exames avaliados às cegas por 2 radiologistas independentes.
- **Métodos empregados:** Para cada sinal de TC (gás extraluminal, hemoperitônio, ausência/realce moderado da parede intestinal, lesão de órgão sólido, entre outros), calcularam kappa, sensibilidade, especificidade e acurácia. Construíram uma árvore de decisão via **particionamento recursivo** (não é radiômica de textura, mas um modelo de classificação clássico e interpretável sobre variáveis categóricas de achados radiológicos).
- **Principais métricas/resultados:** Sinais com kappa > 0,6: gás extraluminal, hemoperitônio, realce ausente/moderado da parede intestinal, lesão de órgão sólido. A árvore final (gás extraluminal + realce ausente/moderado da parede) atingiu 86% sensibilidade / 96% especificidade no treino e 92% sensibilidade / 88% especificidade na validação.
- **Limitações apontadas pelos próprios autores:** Amostra pequena (79 + 37 pacientes); estudo retrospectivo, single/poucos centros; dependência de leitura e classificação manual dos sinais de TC pelos radiologistas (não há automação/extração quantitativa de imagem).
- **O que aproveitamos para o nosso baseline:** É a referência direta para a classe **bowel/mesenteric injury** do nosso desafio (o órgão com menor prevalência de lesão na nossa amostra — 2,6%). Mostra que sinais interpretáveis simples (gás extraluminal, realce de parede) têm alto poder discriminativo. **Achado convergente no Marco 3:** para `bowel`, foi justamente o descritor de **intensidade** (não textura nem forma) combinado ao SVM linear que obteve o melhor resultado do projeto para esse órgão (0,724 de balanced accuracy, AUC 0,668) — coerente com a ideia de Lansier et al. de que sinais relativamente simples de intensidade/atenuação (gás, realce) carregam a maior parte do sinal discriminativo para lesão intestinal, mais do que padrões de textura complexos. Vale citar esse paralelo na Discussão.

---

## 5. Hemoperitoneum Detection on Non-Contrast CT: Radiomics-Driven Explainable ML

- **Autores / Ano:** Bhagawati R, Hazarika S, Chanda S. (2026)
- **Link / DOI:** https://doi.org/10.1016/j.bspc.2025.109177 — *Biomedical Signal Processing and Control*, 113(Part C):109177
- **Objetivo do artigo:** Propor o "BoostFusion", um framework de ML radiômico e explicável para detectar hemoperitônio (achado ligado ao alvo `extravasation` do nosso desafio) em TC **sem contraste**, eliminando a dependência de contraste intravenoso.
- **Dataset usado:** 7.015 cortes de TC sem contraste de 52 pacientes de trauma, estudo retrospectivo de instituição única, anotação manual para presença de hemoperitônio.
- **Métodos empregados:** Pipeline radiômico estruturado (features de primeira ordem, GLCM, GLRLM, GLSZM, GLDM, NGTDM) + redução de dimensionalidade e seleção de features + validação cruzada estratificada de 20 folds. Testaram 9 modelos: Regressão Logística, Gaussian Naive Bayes, LDA, SVM, Árvore de Decisão, Random Forest, XGBoost, LightGBM e o modelo proposto BoostFusion (ensemble de gradient boosting). Usaram SHAP para explicabilidade.
- **Principais métricas/resultados:** BoostFusion atingiu sensibilidade média de 96%, especificidade 96%, F1 96%, AUC-ROC 99%, MCC 93% — métrica composta própria (Comprehensive Classification Score, CCS) combinando essas medidas.
- **Limitações apontadas pelos próprios autores:** Estudo retrospectivo de instituição única (baixa generalização multicêntrica); amostra pequena em nível de paciente (52); não usa TC contrastada (compromisso proposital para cenários de baixo recurso, mas reduz sensibilidade a sangramento ativo comparado ao padrão-ouro).
- **O que aproveitamos para o nosso baseline:** Repertório completo de radiômica de textura (GLCM/GLRLM/GLSZM/GLDM/NGTDM) alinhado ao que o enunciado do TP1 recomenda (nossa extração de textura no Marco 3 usa GLCM, um subconjunto desse repertório); metodologia de validação cruzada estratificada com múltiplos classificadores clássicos comparados lado a lado — molde direto para nossa grade de experimentos descritor × modelo da Semana 3; relevante especificamente para o alvo `extravasation`, que na nossa amostra é o segundo mais raro (7,0% dos pacientes) — nosso melhor resultado aqui (forma + XGBoost, 0,589) ficou bem abaixo do BoostFusion, esperado dada a diferença de escala e de segmentação prévia do órgão.

---

## 6. The Image Biomarker Standardization Initiative (IBSI)

- **Autores / Ano:** Zwanenburg A, Vallières M, Abdalah MA, et al. (2020)
- **Link / DOI:** https://doi.org/10.1148/radiol.2020191145 — *Radiology*, 295(2):328–338
- **Objetivo do artigo:** Padronizar definições e valores de referência para 174 características radiômicas, permitindo verificação e calibração de diferentes softwares de radiômica.
- **Dataset usado:** Não é um estudo clínico — usa um phantom digital (fase I), uma TC de um paciente com câncer de pulmão (fase II) e uma coorte pública de 51 pacientes com sarcoma de partes moles, multimodalidade (fase III), para validar reprodutibilidade.
- **Métodos empregados:** Processo iterativo de consenso entre 25 equipes de pesquisa com implementações próprias de software, comparando valores calculados para cada feature (primeira ordem, forma, GLCM, GLRLM, GLSZM, GLDM, NGTDM) até convergência. Reprodutibilidade final avaliada por coeficiente de correlação intraclasse (ICC).
- **Principais métricas/resultados:** 169 das 174 features padronizadas com consenso forte ou muito forte; reprodutibilidade excelente (ICC > 0,90) para 166/174 (CT), 164/174 (PET) e 164/174 (MRI) das features padronizadas.
- **Limitações apontadas pelos próprios autores:** Não avaliaram fractais nem todos os filtros de imagem; parâmetros de aquisição/reconstrução/segmentação continuam sendo fontes de variabilidade não cobertas pelo estudo; a reprodutibilidade validada não garante robustez em cenários multicêntricos/multi-scanner.
- **O que aproveitamos para o nosso baseline:** É a referência metodológica que justifica *como* extrair radiômica de forma correta e comparável — nomenclatura padronizada (GLCM, GLRLM, GLSZM, GLDM, NGTDM) e o fluxo de processamento que seguimos na extração de textura GLCM do Marco 3 (é a referência que o próprio enunciado do TP1 cita ao pedir radiômica "IBSI-compatível").

---

## 7. PyRadiomics — Computational Radiomics System to Decode the Radiographic Phenotype

- **Autores / Ano:** van Griethuysen JJM, Fedorov A, Parmar C, et al. (2017)
- **Link / DOI:** https://doi.org/10.1158/0008-5472.CAN-17-0339 — *Cancer Research*, 77(21):e104–e107
- **Objetivo do artigo:** Apresentar o PyRadiomics, uma plataforma open-source em Python para extração padronizada de características radiômicas de imagens médicas.
- **Dataset usado:** Estudo de caso com 429 lesões pulmonares de 302 pacientes (Lung Image Database Consortium), com 4 segmentações manuais por lesão.
- **Métodos empregados:** Pipeline de 4 etapas — carregamento/pré-processamento, filtros (wavelet, Laplacian of Gaussian, square, logarithm etc.), cálculo de features (primeira ordem, forma, GLCM, GLRLM, GLSZM) e retorno estruturado dos resultados. Avaliaram estabilidade das features por ICC entre as 4 segmentações e treinaram um classificador Random Forest para discriminar malignidade.
- **Principais métricas/resultados:** Alta estabilidade (ICC > 0,8) para features de primeira ordem, textura e LoG; biomarcador multivariado (25 features selecionadas + Random Forest) atingiu AUC de 0,79 [0,73–0,85] na coorte de validação.
- **Limitações apontadas pelos próprios autores:** Features de forma e wavelet mostraram estabilidade só moderada (sensíveis à variabilidade de delineação/segmentação manual); resultados do estudo de caso são específicos para nódulos pulmonares, não generalizáveis a outras aplicações sem validação própria.
- **O que aproveitamos para o nosso baseline:** Usamos `scikit-image` (GLCM via `graycomatrix`/`graycoprops`) em vez do PyRadiomics propriamente dito no Marco 3, mas a lógica de organização por famílias de descritores (primeira ordem/intensidade, forma, textura) segue a mesma estrutura que este artigo descreve. A ressalva sobre estabilidade moderada de features de forma é coerente com o que observamos: `shape_*` (momentos de Hu) teve peso menor que as features de intensidade na importância calculada para `liver` no Marco 3 — relevante para nossa decisão de reportar essa limitação na Discussão.

---

## 8. Future Perspectives on Radiomics in Acute Liver Injury and Liver Trauma

- **Autores / Ano:** Brunese MC, Avella P, Cappuccio M, et al. (2024)
- **Link / DOI:** https://doi.org/10.3390/jpm14060572 — *Journal of Personalized Medicine*, 14(6):572
- **Objetivo do artigo:** Revisão sistemática sobre o uso de IA/radiômica para detectar e quantificar áreas de lesão hepática aguda (a maioria por trauma) em adultos e crianças.
- **Dataset usado:** Não é um estudo original — revisão de 6 estudos (564 pacientes no total: 170 crianças, 394 adultos) selecionados no PubMed, publicados entre 2018–2023.
- **Métodos empregados:** Análise de literatura seguindo critérios de inclusão (≥10 pacientes, artigos originais); 5 dos 6 estudos usaram TC, 1 usou FAST-Ultrassom; a maioria dos estudos revisados usa a escala AAST (I–VI para fígado) como referência de gravidade — a mesma lógica de baixo/alto grau usada no nosso desafio.
- **Principais métricas/resultados:** Alta performance diagnóstica reportada nos estudos revisados; 3 dos 6 estudos com especificidade > 80%.
- **Limitações apontadas pelos próprios autores:** Poucos estudos disponíveis (só 6 atenderam aos critérios), cohorts pequenas, falta de validação em coortes maiores; segmentação manual ainda é a abordagem dominante na literatura revisada, não automatizada.
- **O que aproveitamos para o nosso baseline:** Traz o detalhamento da escala AAST especificamente para fígado (graus I–VI, com critérios de % de área de superfície, hematoma, profundidade, ruptura de veia cava) — útil para explicar no artigo por que a divisão baixo/alto grau (usada no RATIC) simplifica uma escala mais granular. Também reforça, como panorama de literatura, que segmentação manual/semi-automática ainda domina esse campo, contextualizando a dificuldade adicional que enfrentamos ao propor um pipeline mais automatizado, sem segmentação de órgão — o que ajuda a explicar por que nosso melhor resultado para `liver` (0,456 de balanced accuracy) fica longe dos números da literatura clínica.

---

## 9. Radiomics-Based Machine Learning for Splenic Injury Diagnosis Using CT Images

- **Autores / Ano:** Alimiri Dehbaghi H, Khoshgard K, Jafari S. (2026)
- **Link / DOI:** https://doi.org/10.1038/s41598-026-61595-3 — *Scientific Reports* (Article in Press, versão ainda não editorada — citar com essa ressalva)
- **Objetivo do artigo:** Avaliar o impacto de modelos de ML e features radiômicas no diagnóstico de lesões traumáticas esplênicas em TC — mesmo grupo de autores e mesma abordagem metodológica do artigo 3 do nosso fichamento (que tratava do fígado).
- **Dataset usado:** 600 imagens de TC (mesma origem: dataset Kaggle RSNA 2023 Abdominal Trauma Detection), incluindo baço saudável, com lesão leve e com lesão grave.
- **Métodos empregados:** Mesma metodologia do artigo do fígado (mesmo grupo): segmentação manual, extração radiômica (features de primeira ordem + GLCM/GLDM/GLRLM/GLSZM/NGTDM com decomposições wavelet), comparação entre modelos clássicos de ML.
- **Principais métricas/resultados:** Ainda não temos o texto completo dos resultados numéricos nesta versão "in press" da submissão, mas a abordagem espelha o artigo 3 (LGBM/XGBoost/Ridge com métricas de acurácia, AUC, sensibilidade, especificidade por grau de gravidade).
- **Limitações apontadas pelos próprios autores:** Versão "Article in Press" — texto ainda não editorado, sujeito a correções antes da publicação final; mesmas limitações estruturais do artigo-irmão do fígado (segmentação manual, sem informação clínica do paciente).
- **O que aproveitamos para o nosso baseline:** É o **par direto** do artigo 3 do nosso fichamento, agora cobrindo o baço em vez do fígado, com o mesmo dataset do nosso desafio — junto, os dois formam a comparação mais próxima possível ao que estamos construindo. Nosso melhor resultado para `spleen` no Marco 3 (textura + SVM, 0,437 de balanced accuracy) serve de ponto de comparação direto quando a versão final deste artigo publicar seus números.

---

## 10. RSNA 2023 Abdominal Trauma AI Challenge: Review and Outcomes

- **Autores / Ano:** Hermans S, Hu Z, Ball RL, et al. (2025)
- **Link / DOI:** https://doi.org/10.1148/ryai.240334 — *Radiology: Artificial Intelligence*, 7(1):e240334
- **Objetivo do artigo:** Avaliar retrospectivamente, com métricas clinicamente interpretáveis, o desempenho dos 8 modelos vencedores do desafio RSNA 2023 Abdominal Trauma Detection (o mesmo desafio do nosso TP1) — diferente do artigo 1 do nosso fichamento (Rudie et al.), que descreve a construção do *dataset*; este descreve o desempenho dos *modelos vencedores*.
- **Dataset usado:** O mesmo RATIC (4.274 exames), usando especificamente o conjunto de teste privado (723 exames) para reavaliar os 8 modelos premiados.
- **Métodos empregados:** Reavaliação binária (normal vs. lesionado) e também para alto grau, usando AUC, acurácia, sensibilidade, especificidade, valor preditivo positivo/negativo e F1, com thresholds ótimos definidos pelo índice de Youden no conjunto de teste público.
- **Principais métricas/resultados — benchmark direto para o nosso baseline:** AUC médio de 0,92 (fígado), 0,91 (baço) e 0,94 (rim) para detecção binária; AUC de 0,98 para alto grau nos três órgãos sólidos; AUC de 0,85 tanto para lesão intestinal/mesentérica quanto para extravasamento ativo — os alvos mais difíceis, confirmando o que já intuímos pela EDA (essas são as classes mais raras e com sinal mais sutil).
- **Limitações apontadas pelos próprios autores:** Modelos otimizados por uma métrica composta da competição, não por tarefa individual (poderia haver ganho ao otimizar cada alvo separadamente); dataset não inclui fase tardia de imagem; alguns rótulos de referência tinham taxa de erro maior em um dos centros contribuintes (corrigidos via revisão por 3 radiologistas); modelos ainda distantes de deployment clínico real (exigem validação prospectiva externa).
- **O que aproveitamos para o nosso baseline:** É o **benchmark mais importante do nosso projeto** — dá o teto de desempenho state-of-the-art (com deep learning, o que nosso TP1 não pode usar) para cada um dos 5 alvos.

  **Comparação direta com os resultados do Marco 3** (teste com 100 pacientes, métodos clássicos):

  | Órgão | Nosso AUC-ROC (melhor combo) | AUC Hermans et al. (deep learning) | Distância do teto |
  |---|---|---|---|
  | kidney | **0,888** (combinado + SVM) | 0,94 | −0,05 |
  | spleen | 0,609 (textura + SVM) | 0,91 | −0,30 |
  | liver | 0,565 (intensidade + RandomForest) | 0,92 | −0,36 |
  | bowel | 0,668 (intensidade + SVM) | 0,85 | −0,18 |
  | extravasation | 0,550 (forma + XGBoost) | 0,85 | −0,30 |

  `kidney` é, de longe, o órgão em que nosso baseline clássico mais se aproximou do teto de deep learning — um resultado forte para destacar na seção de Resultados. Os demais órgãos ficam mais distantes, especialmente `liver` e `spleen`, o que é esperado e defensável: sem segmentação prévia de órgão (diferente dos modelos vencedores da competição, que usam pipelines de deep learning com segmentação implícita/explícita), nosso pipeline 2.5D com pooling por paciente perde resolução espacial que provavelmente é decisiva para esses dois órgãos. É exatamente esse contraste que estabelece o "piso metodológico" que a Seção 1 do enunciado do TP1 pede que o baseline demonstre.

---

## Síntese comparativa (para a seção "Trabalhos Relacionados" do artigo)

A literatura recente sobre trauma abdominal por TC converge fortemente no uso de **radiômica de textura clássica** (GLCM, GLRLM, GLSZM, GLDM, NGTDM) combinada a classificadores de gradient boosting (XGBoost, LightGBM) ou ensembles (Random Forest) — é o padrão nos artigos 3, 5 e 9, e também aparece no artigo 2 (Gabor/fractal/forma). Os artigos 6 e 7 (IBSI e PyRadiomics) fornecem a base metodológica e a ferramenta prática para que essa extração seja padronizada e reprodutível — algo que o próprio enunciado do TP1 exige explicitamente ("radiômica IBSI-compatível").

Esses trabalhos aplicados, no entanto, são de **órgão único** (baço, fígado, ou apenas extravasamento) e de **amostras pequenas ou balanceadas por construção** (52 a 600 imagens/pacientes), diferente do escopo do nosso TP1, que precisa lidar com 5 alvos simultâneos e desbalanceamento real (não corrigido por desenho amostral) no mesmo pipeline. Os artigos 3 e 9, do mesmo grupo de autores, ilustram bem isso: usam exatamente o nosso dataset, mas tratam fígado e baço como problemas independentes e com classes balanceadas artificialmente (200/200/200), sem a integração multi-órgão nem o desbalanceamento real que o desafio original (e nosso TP1) exige.

Há uma divergência metodológica relevante: o artigo 2 (Wang et al.) demonstra empiricamente que métodos clássicos com features hand-crafted **superam** uma abordagem de deep learning (ResNet+LSTM) quando o dataset é pequeno — achado que justifica diretamente a restrição do nosso TP1 (proibição de redes profundas) como escolha metodológica defensável, não apenas como regra artificial da disciplina. Por outro lado, o artigo 10 (Hermans et al.) mostra que os modelos vencedores da competição original, todos baseados em deep learning, atingem AUCs de até 0,98 — um teto de desempenho que nosso baseline clássico não deve almejar igualar, mas contra o qual pode se posicionar explicitamente como "piso metodológico" (exatamente a linguagem usada na Seção 1 do enunciado do TP1).

O artigo 4 (Lansier et al.) foge do padrão radiômico — usa achados semânticos de TC (gás extraluminal, realce de parede) com um modelo de árvore de decisão simples, mostrando que, para o alvo mais raro do nosso desafio (bowel/mesenteric, 2,6% de prevalência), talvez valha a pena complementar as features de textura com features de região/forma mais diretamente ligadas aos sinais radiológicos clássicos. O artigo 8 (Brunese et al., revisão sobre fígado) reforça esse ponto de forma mais ampla: a literatura de radiômica em trauma abdominal ainda depende fortemente de segmentação manual, uma limitação que nosso pipeline (com crop automático de corpo, mas sem segmentação de órgão ainda) também compartilha e deve discutir.

### Atualização após o Marco 3 (grade de experimentos)

Com a grade de experimentos concluída (5 órgãos × 3 famílias de descritores + combinado × 3 modelos clássicos, validação cruzada, 500 pacientes), alguns pontos da síntese acima podem ser confirmados ou refinados empiricamente com dados próprios, não só por analogia com a literatura:

- O achado do artigo 2 (Wang et al.) de que métodos clássicos competem bem com deep learning em amostra pequena **não se estendeu igualmente entre os métodos clássicos entre si**: em nossa grade, o SVM superou consistentemente Random Forest e XGBoost nos alvos mais raros, mesmo depois de reponderar as três abordagens para desbalanceamento. Isso qualifica a recomendação "classificador clássico + hand-crafted features" da literatura — importa também *qual* classificador clássico, não só a família (radiômica vs. deep learning).
- A hipótese de Lansier et al. (artigo 4), de que sinais relativamente simples carregam a maior parte do sinal discriminativo para `bowel`, encontrou eco direto: intensidade (não textura) foi o melhor descritor isolado para esse órgão em nossos experimentos.
- O teto de Hermans et al. (artigo 10) foi alcançado de forma bem mais próxima para `kidney` (AUC 0,888 vs. 0,94) do que para `liver`/`spleen`/`extravasation` — reforçando a hipótese, discutida nos artigos 8 e 9, de que a ausência de segmentação de órgão pesa mais nesses casos.
- O achado de que "combinar todos os descritores nem sempre ajuda" (visto no estudo de ablação do Marco 3, para `kidney` e `liver`) não tem um paralelo direto nos 10 artigos revisados — a maioria trabalha com um número menor de features por já ter segmentação prévia do órgão, reduzindo ruído. Isso pode ser posicionado como uma contribuição específica da nossa análise na seção de Discussão: sem segmentação, a seleção de descritores por família pode ser mais importante do que simplesmente combinar tudo.

Lacuna que nosso baseline ainda pode discutir: nenhum dos artigos aplicados (2, 3, 4, 5, 9) lida com os **5 alvos simultaneamente e a métrica composta ponderada** que o desafio RSNA original exige (artigos 1 e 10) — é justamente aí que mora a dificuldade adicional do nosso trabalho em relação à literatura existente, tanto clássica quanto a de deep learning.
