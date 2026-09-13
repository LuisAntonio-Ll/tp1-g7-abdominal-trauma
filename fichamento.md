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
- **O que aproveitamos para o nosso baseline:** É a referência primária — descreve exatamente os arquivos (`train_2024.csv` etc.) que estamos usando, confirma a divisão baixo/alto grau (AAST I–III vs IV–V) que orienta a formulação do problema, e documenta o desbalanceamento que já vimos na EDA (ex: extravasamento em apenas ~336/4274 casos).

---

## 2. Automated Spleen Injury Detection Using 3D Active Contours and Machine Learning

- **Autores / Ano:** Wang J, Wood A, Gao C, Najarian K, Gryak J. (2021)
- **Link / DOI:** https://doi.org/10.3390/e23040382 — *Entropy*, 23(4):382
- **Objetivo do artigo:** Propor um método automatizado para detectar lacerações esplênicas em TC usando segmentação por contornos ativos 3D seguida de classificação com features hand-crafted.
- **Dataset usado:** 99 exames de TC (Michigan Medicine + dataset público CIREN), 54 baços saudáveis e 45 lacerados, classificados pela escala AAST/AIS (graus I–V).
- **Métodos empregados:** Segmentação automática do baço (método próprio anterior, active contours). Extração de 4 famílias de features: histograma (entropia de Rényi, média, variância, assimetria, curtose), fractal (dimensão fractal por box-counting), Gabor (banco de 40 filtros, 5 escalas × 8 orientações) e forma (circularidade, excentricidade, orientação, diferença área/área convexa). Classificadores: Random Forest, Naive Bayes, SVM, k-NN ensemble, subspace discriminant ensemble — todos com validação cruzada de 5 folds. Comparação direta contra um pipeline de deep learning (ResNet-50 + LSTM).
- **Principais métricas/resultados:** Random Forest foi o melhor classificador clássico: AUC 0,91, F1 0,80 no teste. **Superou a abordagem de deep learning** (AUC 0,72), atribuído ao tamanho pequeno da amostra — achado direto que embasa a escolha metodológica do nosso próprio TP1 (métodos clássicos como baseline robusto em cenário de dados limitados).
- **Limitações apontadas pelos próprios autores:** Erros de classificação concentrados em lesões leves/moderadas (AIS 2–3), não nas graves; segmentações imperfeitas (6 de 99 casos descartados por erro de segmentação); dataset pequeno por grau de gravidade individual; extensão para multi-classe (por grau) e para outros tipos de lesão esplênica (hematomas, hemorragias) deixada para trabalhos futuros.
- **O que aproveitamos para o nosso baseline:** Repertório de descritores (Gabor, fractal, forma, histograma) aplicável ao baço/fígado/rim; validação empírica de que RF com features hand-crafted pode superar deep learning em dataset pequeno — argumento direto para a Introdução do nosso artigo; metodologia de partição por paciente e CV de 5 folds, igual à exigida no nosso protocolo.

---

## 3. Diagnosis of Traumatic Liver Injury on CT Using Machine Learning Algorithms and Radiomics Features

- **Autores / Ano:** Alimiri Dehbaghi H, Khoshgard K, Sharini H, Khairabadi SJ. (2024)
- **Link / DOI:** https://doi.org/10.4103/jrms.jrms_847_23 — *Journal of Research in Medical Sciences*, 29:77
- **Objetivo do artigo:** Avaliar o desempenho de modelos de ML combinados a features radiômicas para diagnosticar e estadiar (leve/grave) lesão hepática traumática em TC.
- **Dataset usado:** 600 cortes axiais de TC do próprio dataset Kaggle RSNA 2023 Abdominal Trauma Detection (200 fígado saudável, 200 lesão leve, 200 lesão grave) — **mesmo dataset do nosso desafio**, usado em recorte 2D.
- **Métodos empregados:** Segmentação manual no 3D Slicer. Extração radiômica via toolbox do 3D Slicer: features de primeira ordem + textura (GLCM, GLDM, GLRLM, GLSZM, NGTDM), aplicadas sobre 8 decomposições wavelet (HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL). 30 modelos testados inicialmente, reduzidos a 3: LightGBM, Ridge Classifier, XGBoost. Split 75/25 treino/teste.
- **Principais métricas/resultados:** LightGBM foi o melhor: acurácia 94% (lesão leve) e 99% (lesão grave), AUC 93,8% e 99,0%, respectivamente. Tabela 1 do artigo lista as features radiômicas mais importantes por modelo — referência direta para nossa etapa de seleção de características.
- **Limitações apontadas pelos próprios autores:** Falta de acesso a informações clínicas do paciente (só imagem); segmentação manual (não automatizada); sugerem aumentar o número de sujeitos em trabalhos futuros.
- **O que aproveitamos para o nosso baseline:** É o artigo mais diretamente comparável — usa o mesmo dataset, radiômica IBSI-compatível (recomendada no enunciado do TP1) e os mesmos 3 classificadores clássicos que pretendemos testar. Serve de ponto de comparação numérico direto (nosso baseline vs. os resultados desse estudo) e de checklist de quais features GLCM/GLDM/GLRLM/GLSZM/NGTDM priorizar.

---

## 4. Algoritmo diagnóstico baseado em TC para identificar lesões intestinais e/ou mesentéricas em trauma abdominal fechado

- **Autores / Ano:** Lansier A, Bourillon C, Cuénod CA, et al. (2023)
- **Link / DOI:** https://doi.org/10.1007/s00330-022-09200-9 — *European Radiology*, 33(3):1918–1927
- **Objetivo do artigo:** Desenvolver e validar um algoritmo diagnóstico baseado em sinais de TC (não radiômica/ML de imagem, mas achados semânticos) para lesão intestinal e/ou mesentérica em trauma abdominal fechado.
- **Dataset usado:** Coorte de treinamento com 79 pacientes (29 com lesão intestinal/mesentérica, 50 sem) e coorte de validação com 37 pacientes (13 com lesão, 24 sem). Exames avaliados às cegas por 2 radiologistas independentes.
- **Métodos empregados:** Para cada sinal de TC (gás extraluminal, hemoperitônio, ausência/realce moderado da parede intestinal, lesão de órgão sólido, entre outros), calcularam kappa, sensibilidade, especificidade e acurácia. Construíram uma árvore de decisão via **particionamento recursivo** (não é radiômica de textura, mas um modelo de classificação clássico e interpretável sobre variáveis categóricas de achados radiológicos).
- **Principais métricas/resultados:** Sinais com kappa > 0,6: gás extraluminal, hemoperitônio, realce ausente/moderado da parede intestinal, lesão de órgão sólido. A árvore final (gás extraluminal + realce ausente/moderado da parede) atingiu 86% sensibilidade / 96% especificidade no treino e 92% sensibilidade / 88% especificidade na validação.
- **Limitações apontadas pelos próprios autores:** Amostra pequena (79 + 37 pacientes); estudo retrospectivo, single/poucos centros; dependência de leitura e classificação manual dos sinais de TC pelos radiologistas (não há automação/extração quantitativa de imagem).
- **O que aproveitamos para o nosso baseline:** É a referência direta para a classe **bowel/mesenteric injury** do nosso desafio (o órgão com menor prevalência de lesão na nossa EDA — 2,26%). Mostra que sinais interpretáveis simples (gás extraluminal, realce de parede) têm alto poder discriminativo — pode inspirar features de região/forma específicas para essa classe, além de servir de benchmark de sensibilidade/especificidade alvo.

---

## 5. Hemoperitoneum Detection on Non-Contrast CT: Radiomics-Driven Explainable ML

- **Autores / Ano:** Bhagawati R, Hazarika S, Chanda S. (2026)
- **Link / DOI:** https://doi.org/10.1016/j.bspc.2025.109177 — *Biomedical Signal Processing and Control*, 113(Part C):109177
- **Objetivo do artigo:** Propor o "BoostFusion", um framework de ML radiômico e explicável para detectar hemoperitônio (achado ligado ao alvo `extravasation` do nosso desafio) em TC **sem contraste**, eliminando a dependência de contraste intravenoso.
- **Dataset usado:** 7.015 cortes de TC sem contraste de 52 pacientes de trauma, estudo retrospectivo de instituição única, anotação manual para presença de hemoperitônio.
- **Métodos empregados:** Pipeline radiômico estruturado (features de primeira ordem, GLCM, GLRLM, GLSZM, GLDM, NGTDM) + redução de dimensionalidade e seleção de features + validação cruzada estratificada de 20 folds. Testaram 9 modelos: Regressão Logística, Gaussian Naive Bayes, LDA, SVM, Árvore de Decisão, Random Forest, XGBoost, LightGBM e o modelo proposto BoostFusion (ensemble de gradient boosting). Usaram SHAP para explicabilidade.
- **Principais métricas/resultados:** BoostFusion atingiu sensibilidade média de 96%, especificidade 96%, F1 96%, AUC-ROC 99%, MCC 93% — métrica composta própria (Comprehensive Classification Score, CCS) combinando essas medidas.
- **Limitações apontadas pelos próprios autores:** Estudo retrospectivo de instituição única (baixa generalização multicêntrica); amostra pequena em nível de paciente (52); não usa TC contrastada (compromisso proposital para cenários de baixo recurso, mas reduz sensibilidade a sangramento ativo comparado ao padrão-ouro).
- **O que aproveitamos para o nosso baseline:** Repertório completo de radiômica de textura (GLCM/GLRLM/GLSZM/GLDM/NGTDM) alinhado ao que o enunciado do TP1 recomenda; metodologia de validação cruzada estratificada com múltiplos classificadores clássicos comparados lado a lado — modelo direto para nossa "grade de experimentos descritor × modelo" da Semana 3; relevante especificamente para o alvo `extravasation`, que na nossa EDA é o segundo mais raro (6,83% dos pacientes).

---

## Síntese comparativa (para a seção "Trabalhos Relacionados" do artigo)

A literatura recente sobre trauma abdominal por TC converge fortemente no uso de **radiômica de textura clássica** (GLCM, GLRLM, GLSZM, GLDM, NGTDM) combinada a classificadores de gradient boosting (XGBoost, LightGBM) ou ensembles (Random Forest) — é o padrão nos artigos 3 e 5, e também aparece no artigo 2 (Gabor/fractal/forma). Esses trabalhos, no entanto, são de **órgão único** (baço, fígado, ou apenas extravasamento) e de **amostras pequenas** (52 a 600 imagens/pacientes), diferente do escopo do nosso TP1, que precisa lidar com 5 alvos simultâneos no mesmo pipeline.

Há uma divergência relevante: o artigo 2 (Wang et al.) demonstra empiricamente que métodos clássicos com features hand-crafted **superam** uma abordagem de deep learning (ResNet+LSTM) quando o dataset é pequeno — achado que justifica diretamente a restrição do nosso TP1 (proibição de redes profundas) como escolha metodológica defensável, não apenas como regra artificial da disciplina.

O artigo 4 (Lansier et al.) foge do padrão radiômico — usa achados semânticos de TC (gás extraluminal, realce de parede) com um modelo de árvore de decisão simples, mostrando que, para o alvo mais raro do nosso desafio (bowel/mesenteric, 2,26% de prevalência), talvez valha a pena complementar as features de textura com features de região/forma mais diretamente ligadas aos sinais radiológicos clássicos, em vez de depender só de radiômica genérica.

Lacuna que nosso baseline pode discutir: nenhum dos 4 artigos aplicados (2, 3, 4, 5) lida com os **5 alvos simultaneamente e a métrica composta ponderada** que o desafio RSNA original exige (artigo 1) — é justamente aí que mora a dificuldade adicional do nosso trabalho em relação à literatura existente.
