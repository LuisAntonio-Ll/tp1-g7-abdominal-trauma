# Formulação do problema — G7 (Abdominal Trauma Detection)

**Entrada:** Uma série de TC abdominal/pélvica (formato DICOM) de um paciente adulto,
adquirida em protocolo de trauma — fase portal venosa isolada, bifásica (arterial +
portal venosa) ou split bolus, conforme os protocolos aceitos pelo desafio RATIC
(Rudie et al., 2024). Cada série contém dezenas a centenas de cortes axiais.

**Saída:** Para cada paciente, uma classificação de gravidade em 5 alvos:

| Alvo | Classes | Prevalência observada na amostra (EDA) |
|---|---|---|
| `bowel` (intestino/mesentério) | healthy / injury | 97,74% / 2,26% |
| `extravasation` (extravasamento ativo) | healthy / injury | 93,17% / 6,83% |
| `kidney` (rim) | healthy / low / high | 93,10% / 4,48% / 2,41% |
| `liver` (fígado) | healthy / low / high | 89,20% / 8,67% / 2,13% |
| `spleen` (baço) | healthy / low / high | 88,18% / 6,67% / 5,15% |

mais um alvo agregado `any_injury` (presença de qualquer lesão), que na amostra
apareceu bem mais equilibrado (~70% sem lesão / ~30% com lesão) do que os alvos
isolados — porque um único paciente pode contribuir para vários órgãos ao mesmo
tempo.

**Unidade de amostra:** **Paciente** (não corte, não série). Justificativa:
- Os rótulos do `train_2024.csv` são fornecidos em nível de paciente/estudo, não de corte.
- O enunciado do TP1 exige partição por paciente (não por imagem) para evitar
  vazamento de dados — cortes do mesmo exame não podem ficar espalhados entre
  treino e teste.
- Como o volume de TC tem múltiplos cortes por paciente, a estratégia de agregação
  escolhida é **2.5D com pooling por exame**: extraímos features de um subconjunto de
  cortes representativos (ex: corte central da série, ou os cortes correspondentes à
  região de cada órgão) e agregamos (média/máximo) para gerar um vetor de
  características único por paciente. Essa decisão evita o custo computacional de
  processar volumes 3D completos, mantendo a granularidade paciente-alvo exigida
  pelos rótulos disponíveis.

**Por que é um problema difícil:**

- **Desbalanceamento severo por órgão** (ver tabela acima) — confirmado tanto na
  nossa EDA quanto no artigo do próprio dataset (Rudie et al., 2024: apenas 1.316 dos
  4.274 casos totais, ~30,8%, são positivos para alguma lesão). Isso exige métricas
  como AUC-ROC/AUC-PR em vez de acurácia, e tratamento de desbalanceamento
  (pesos de classe, reamostragem) na modelagem.
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
  mais difícil do nosso desafio, tanto pela raridade (2,26%) quanto pela natureza do
  achado.
- **Teto de desempenho conhecido, mas fora do nosso alcance metodológico** — Hermans
  et al. (2025) reportam AUC médio de 0,92 (fígado), 0,91 (baço) e 0,94 (rim) para
  detecção binária, e 0,85 tanto para `bowel` quanto para `extravasation`, usando os
  modelos vencedores da competição original — todos baseados em deep learning, que
  o nosso TP1 proíbe. Esses números servem de referência de "teto" na discussão de
  resultados, não de meta a bater, e confirmam que `bowel` e `extravasation` são
  objetivamente os alvos mais difíceis do desafio, mesmo para os melhores modelos
  possíveis.
