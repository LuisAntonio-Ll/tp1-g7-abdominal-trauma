# Formulação do problema — G7 (Abdominal Trauma Detection)

**Entrada:** Uma série de TC abdominal/pélvica (formato DICOM) de um paciente adulto,
adquirida em protocolo de trauma — fase portal venosa isolada, bifásica (arterial +
portal venosa) ou split bolus, conforme os protocolos aceitos pelo desafio RATIC
(Rudie et al., 2024). Cada série contém dezenas a centenas de cortes axiais.

**Saída:** Para cada paciente, uma classificação de gravidade em 5 alvos:

| Alvo | Classes | Prevalência observada na amostra (N=500, Marco 3) |
|---|---|---|
| `bowel` (intestino/mesentério) | healthy / injury | 97,40% / 2,60% |
| `extravasation` (extravasamento ativo) | healthy / injury | 93,00% / 7,00% |
| `kidney` (rim) | healthy / low / high | 93,00% / 4,00% / 3,00% |
| `liver` (fígado) | healthy / low / high | 90,80% / 8,20% / 1,00% |
| `spleen` (baço) | healthy / low / high | 88,20% / 5,60% / 6,20% |

mais um alvo agregado `any_injury` (presença de qualquer lesão), que na amostra
apareceu bem mais equilibrado (71,0% sem lesão / 29,0% com lesão) do que os alvos
isolados — porque um único paciente pode contribuir para vários órgãos ao mesmo
tempo.

> **Nota de versão:** a amostra foi ampliada de 180 para **500 pacientes** entre o Marco 2 e
> o Marco 3, mantendo a mesma estratégia de estratificação por `any_injury` e a mesma
> semente (`SEED=42`). O motivo foi metodológico: com apenas 36 pacientes de teste (180
> no total), algumas classes raras (ex. `liver_high`, `kidney_high`) tinham poucos ou
> nenhum representante no conjunto de teste, tornando algumas métricas (AUC-ROC
> macro para alvos de 3 classes) instáveis ou até indefinidas em certas rodadas. Com 500
> pacientes (400 treino / 100 teste), as proporções por classe se mantiveram muito
> próximas às observadas na amostra menor — o desbalanceamento é uma característica do
> problema, não um artefato da amostragem — mas as estimativas de desempenho ficaram
> visivelmente mais estáveis entre validação cruzada e teste.

**Unidade de amostra:** **Paciente** (não corte, não série). Justificativa:
- Os rótulos do `train_2024.csv` são fornecidos em nível de paciente/estudo, não de corte.
- O enunciado do TP1 exige partição por paciente (não por imagem) para evitar
  vazamento de dados — cortes do mesmo exame não podem ficar espalhados entre
  treino e teste.
- Como o volume de TC tem múltiplos cortes por paciente, a estratégia de agregação
  escolhida é **2.5D com pooling por exame**: extraímos features de um subconjunto de
  cortes representativos (5 cortes igualmente espaçados ao longo da série) e agregamos
  por média para gerar um vetor de características único por paciente, cobrindo agora
  **3 famílias de descritores** — intensidade/histograma, textura (GLCM) e forma
  (momentos de Hu e propriedades geométricas) — conforme o mínimo exigido pelo §4.2 do
  enunciado. Essa decisão evita o custo computacional de processar volumes 3D completos,
  mantendo a granularidade paciente-alvo exigida pelos rótulos disponíveis.

**Por que é um problema difícil:**

- **Desbalanceamento severo por órgão** (ver tabela acima) — confirmado tanto na
  nossa EDA quanto no artigo do próprio dataset (Rudie et al., 2024: apenas 1.316 dos
  4.274 casos totais, ~30,8%, são positivos para alguma lesão). Isso exige métricas
  como AUC-ROC/AUC-PR em vez de acurácia, e tratamento de desbalanceamento
  (pesos de classe, reamostragem) na modelagem. Na prática do Marco 3, esse
  desbalanceamento também se mostrou um fator determinante na escolha do **modelo**, não
  só do descritor: o SVM, que reponderamento na margem de decisão
  (`class_weight="balanced"`), lidou melhor com classes abaixo de ~5% de prevalência
  (ex. `bowel_injury`, 2,6%) do que Random Forest e XGBoost, cujo reponderamento agindo
  na escolha dos splits/na função de perda não foi suficiente sozinho para evitar
  colapso para a classe majoritária nos casos mais raros — só passou a funcionar de
  forma consistente depois de reponderar explicitamente as amostras
  (`sample_weight`) no ajuste desses modelos.
- **Múltiplos órgãos e gravidades simultâneas por paciente** — um mesmo exame pode
  ter lesão em vários órgãos ao mesmo tempo, cada um com sua própria escala de
  gravidade (AAST I–III = baixo grau, IV–V = alto grau, conforme Rudie et al.), exigindo
  um pipeline que produza 5 saídas correlacionadas em vez de uma única classificação
  binária. Vale notar que a escala AAST original tem, para o fígado, 6 graus (I–VI —
  Brunese et al., 2024); o RATIC simplifica essa granularidade em baixo/alto grau,
  uma decisão de design do próprio dataset que reduz a resolução do rótulo
  disponível para nós.
- **Necessidade de localização prévia de órgãos** — diferente de um problema de
  classificação de imagem única, aqui a característica relevante está numa região
  específica do volume (o fígado, o baço, cada rim); sem segmentação/localização
  prévia, os descritores de textura/forma captam ruído de estruturas irrelevantes.
  A literatura correlata (Wang et al., 2021, para baço) resolve isso com segmentação
  automática antes da extração de features — abordagem que pretendemos adaptar.
- **Sinal sutil para o alvo mais raro (`bowel`)** — o artigo de Lansier et al. (2023)
  mostra que mesmo especialistas humanos dependem de sinais indiretos (gás
  extraluminal, realce de parede) para esse tipo de lesão, reforçando que é a classe
  mais difícil do nosso desafio, tanto pela raridade (2,6% na amostra atual)
  quanto pela natureza do achado. No Marco 3, `bowel` foi justamente o alvo onde o
  descritor de **intensidade** (não textura ou forma) combinado ao SVM linear obteve o
  melhor resultado do órgão — um achado que vale ser cruzado com os sinais semânticos
  descritos por Lansier et al. na seção de Discussão do artigo.
- **Combinar todos os descritores nem sempre ajuda** — achado do próprio estudo de
  ablação do Marco 3: para `kidney` e `liver`, uma família de descritor isolada superou
  a combinação de todas as três. Com ~400 pacientes de treino e até 28 features
  combinadas, isso é consistente com um efeito de alta dimensionalidade relativa ao
  tamanho da amostra — mais features nem sempre ajuda quando os dados são poucos,
  especialmente para as classes mais raras.
- **Teto de desempenho conhecido, mas fora do nosso alcance metodológico** — Hermans
  et al. (2025) reportam AUC médio de 0,92 (fígado), 0,91 (baço) e 0,94 (rim) para
  detecção binária, e 0,85 tanto para `bowel` quanto para `extravasation`, usando os
  modelos vencedores da competição original — todos baseados em deep learning, que
  o nosso TP1 proíbe. Esses números servem de referência de "teto" na discussão de
  resultados, não de meta a bater, e confirmam que `bowel` e `extravasation` são
  objetivamente os alvos mais difíceis do desafio, mesmo para os melhores modelos
  possíveis. Nosso melhor resultado clássico no Marco 3 chegou a AUC 0,888 em `kidney`
  (combinado + SVM) — próximo do teto de 0,94 reportado por Hermans et al. para esse
  órgão, e um ponto de comparação direto e defensável para a seção de Resultados.
