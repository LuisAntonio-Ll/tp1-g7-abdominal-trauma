# TP1 — Baseline de Projeto de Pesquisa (G7)

**Disciplina:** Tópicos Especiais em Sistemas de Informação
**Professor:** Décio Gonçalves de Aguiar Neto
**Grupo G7:** Luis Antônio Pereira Rabelo · Francisco Ruan da Silva Queiroz · Carlos Daniel da Silva Ribeiro
**Desafio RSNA atribuído:** [2023 — Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

## Objetivo

Construir um baseline com **métodos clássicos de visão computacional e aprendizado de
máquina** (sem redes neurais profundas) para o problema de detecção e classificação de
lesões traumáticas abdominais em exames de TC, usando o dataset do desafio RSNA 2023.

O baseline final cobre os 5 alvos do desafio (`bowel`, `extravasation`, `kidney`, `liver`,
`spleen`), **3 famílias de descritores** (intensidade/histograma, textura GLCM, forma/momentos
de Hu) e **3 modelos clássicos** (SVM, Random Forest, XGBoost), conforme exigido pelo §4.2/§4.3
do enunciado, com seleção de modelo feita **exclusivamente por validação cruzada** (o conjunto
de teste nunca participa de nenhuma decisão de escolha de descritor, modelo ou hiperparâmetro).

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── fichamento.md                          # resumo dos 10 artigos lidos
├── formulacao_problema.md                 # definição da tarefa (entrada/saída/unidade de amostra)
├── Notebooks/
│   ├── marco-1-eda-abdominal-trauma-g7.ipynb
│   ├── marco-2-pre-processing-abdominal-trauma-g7.ipynb
│   ├── marco-3-g7-abdominal-trauma.ipynb       # extração de descritores + grade de experimentos
│   └── marco-4-analise.ipynb                   # recriação dos modelos vencedores + matrizes de confusão
└── Outputs/
    ├── Marco 1 - EDA/
    │   ├── class_distribution_by_organ.png
    │   ├── class_distribution_summary.csv
    │   ├── example_windowing.png
    │   ├── examples_by_class.png
    │   └── sample_patient_ids.csv
    ├── Marco 2 - Pre Processing/
    │   ├── baseline_trivial_results.csv
    │   ├── baseline_vs_first_model.csv
    │   ├── first_model_results.csv             # Regressão Logística (Semana 2)
    │   ├── intensity_features.csv
    │   ├── labeled_features.csv
    │   ├── preprocessing_before_after.png
    │   ├── sample_patient_ids.csv
    │   └── train_test_split.csv
    ├── Marco 3 - Experimentação/
    │   ├── ablation_by_descriptor.csv
    │   ├── ablation_chart.png
    │   ├── baseline_trivial_results.csv
    │   ├── best_combination_per_organ.csv      # seleção final — só por CV, nunca por teste
    │   ├── comparison_baseline_vs_best.csv
    │   ├── feature_importance_top15.png        # bowel, combinado+RF (histórico)
    │   ├── feature_importance_top15_2.png      # bowel, intensidade+RF (melhor RF da grade)
    │   ├── first_model_results.csv
    │   ├── grid_experiments_results.csv        # grade completa: 5 órgãos × 4 descritores × 3 modelos
    │   ├── intensity_features.csv
    │   ├── labeled_features_full.csv
    │   ├── sample_patient_ids.csv
    │   ├── summary_table_descritor_modelo.csv  # tabela descritor × modelo (CV média ± desvio)
    │   └── texture_shape_features.csv
    └── Marco 4 - Análise/
        ├── confusion_matrices.png              # matrizes de confusão dos 5 órgãos (figura do artigo)
        └── predicoes_por_paciente.csv           # classe real vs. prevista por paciente, para análise de erro
```

## Dados

Os dados **não estão neste repositório** (política do próprio Kaggle e do enunciado do
TP1 — dados brutos não vão para o repositório).

- Fonte: [RSNA 2023 Abdominal Trauma Detection — Kaggle](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- É necessário aceitar os termos da competição no Kaggle antes de acessar os dados.
- A amostra estratificada usada no baseline (lista de `patient_id`, estratificada por
  `any_injury`, semente fixa `SEED=42`) está documentada em
  `Outputs/Marco 1 - EDA/sample_patient_ids.csv`.
- **Tamanho da amostra: 500 pacientes** (ampliada a partir dos 180 pacientes usados nas
  primeiras versões do Marco 2/3 — o teste com 36 pacientes se mostrou instável demais para
  algumas classes raras; com 500 pacientes o split fica em 400 treino / 100 teste).
- A partição treino/teste (`train_test_split.csv`) é **congelada** desde a reamostragem —
  não deve ser recalculada nas próximas semanas.

## Como rodar

1. Acesse o Kaggle e crie um **Notebook** vinculado à competição
   `rsna-2023-abdominal-trauma-detection` (o dataset já vem montado automaticamente em
   `/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection`).
2. Rode em sequência:
   - `marco-1-eda-abdominal-trauma-g7.ipynb` (EDA + amostragem estratificada);
   - `marco-2-pre-processing-abdominal-trauma-g7.ipynb` (pipeline de pré-processamento,
     baseline trivial, primeiro modelo — Regressão Logística);
   - `marco-3-g7-abdominal-trauma.ipynb` (extração das 3 famílias de descritores + grade de
     experimentos descritor × modelo, com validação cruzada e ajuste de hiperparâmetros);
   - `marco-4-analise.ipynb` (recria os modelos vencedores de cada órgão — os mesmos
     hiperparâmetros escolhidos por CV no Marco 3 — e gera as matrizes de confusão e as
     previsões por paciente para a análise de erro qualitativa do artigo).
3. Confirme que a variável `DATA_DIR`, na célula de configuração de cada notebook, aponta para:
   ```python
   DATA_DIR = Path("/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection")
   ```
4. **Se cada marco estiver rodando em uma sessão separada do Kaggle** (não é a mesma sessão
   contínua desde o Marco 1): anexe o notebook do marco anterior como **Input** do notebook
   atual (painel direito → "+ Add Input" → aba "Notebooks" → selecionar a versão salva do
   marco anterior). Isso é necessário porque os artefatos "congelados" de um marco
   (`sample_patient_ids.csv`, `train_test_split.csv`, `intensity_features.csv`,
   `texture_shape_features.csv`, `labeled_features_full.csv`, `best_combination_per_organ.csv`
   etc.) precisam estar acessíveis para o marco seguinte reaproveitá-los em vez de
   recalculá-los do zero. Os notebooks já têm uma célula que procura esses arquivos tanto na
   pasta de trabalho local quanto dentro de `/kaggle/input/` (onde ficam os notebooks
   anexados como Input), então não é necessário descobrir o caminho exato manualmente — só
   anexar o Input correto antes de rodar.
5. Rode as células em sequência (Run All). As saídas (gráficos e CSVs) são geradas
   automaticamente na pasta de trabalho do Kaggle e podem ser baixadas em seguida.

> **Nota sobre tempo de execução:** a extração de descritores e a grade de experimentos
> (Marco 3) demoram bem mais que os demais notebooks — a grade completa (5 órgãos × 4
> descritores × 3 modelos, com `GridSearchCV`) leva cerca de 1 hora com 500 pacientes. Já o
> Marco 4 é rápido (segundos) — ele não repete a grade, só retreina os 5 modelos já
> escolhidos como vencedores (um fit por órgão) usando `best_combination_per_organ.csv`.
> Se for só reaproveitar um resultado já salvo (ex.: regenerar uma figura), não é necessário
> rodar a grade inteira de novo — os CSVs de saída já bastam para refazer tabelas e gráficos.

## Reprodutibilidade

- Semente fixa: `SEED = 42` (usada na amostragem estratificada, na partição treino/teste,
  na validação cruzada e em qualquer etapa aleatória).
- Nenhum caminho absoluto pessoal é usado — apenas o caminho padrão de input do Kaggle.
- Desbalanceamento de classes tratado explicitamente: `class_weight="balanced"` (SVM),
  `class_weight="balanced_subsample"` + `sample_weight` (Random Forest) e `sample_weight`
  via `compute_sample_weight("balanced", ...)` (XGBoost, que não tem parâmetro
  `class_weight` nativo).
- Número de dobras da validação cruzada adaptado por alvo
  (k = min(5, n da classe mínima)) — necessário especificamente para `liver_high`, que tem
  só 4 exemplos no treino (k=4; k=5 nos demais órgãos).
- **Seleção de modelo apenas por CV:** a combinação descritor+modelo reportada como
  "melhor" por órgão é escolhida estritamente pela média de validação cruzada — nunca pelo
  resultado no conjunto de teste, para evitar viés de comparação múltipla.
- Dependências declaradas em `requirements.txt`.

## Status do projeto

- [x] Marco 1 — EDA, leitura DICOM, amostragem, fichamento, formulação do problema
- [x] Marco 2 — pipeline de pré-processamento completo, partição por paciente, baseline
      trivial, primeiro modelo (Regressão Logística)
- [x] Marco 3 — extração das 3 famílias de descritores (intensidade, textura GLCM, forma),
      grade de experimentos descritor × modelo (SVM, Random Forest, XGBoost) com validação
      cruzada, estudo de ablação por família de descritor, amostra ampliada para 500
      pacientes
- [x] Marco 4 — análise de erro qualitativa (matrizes de confusão por órgão, previsões por
      paciente), escrita final do artigo (template SBC), teste de reprodutibilidade

### Principais resultados (seleção por CV, teste com 100 pacientes)

| Órgão | Baseline trivial | Combinação selecionada (por CV) | CV (méd ± dp) | Teste (Bal. Acc.) | AUC-ROC |
|---|---|---|---|---|---|
| bowel | 0,500 | combinado + SVM | 0,733 ± 0,135 | 0,352 | 0,286 |
| extravasation | 0,500 | intensidade + SVM | 0,599 ± 0,093 | 0,478 | 0,481 |
| kidney | 0,333 | combinado + SVM | 0,488 ± 0,125 | **0,780** | **0,888** |
| liver | 0,333 | textura + SVM | 0,459 ± 0,134 | **0,356** | 0,573 |
| spleen | 0,333 | intensidade + SVM | 0,478 ± 0,079 | 0,186 | 0,489 |

Sob esse protocolo honesto (seleção só por validação cruzada, nunca por teste), **apenas
kidney e liver superam o baseline trivial no teste**. O rim atinge o melhor resultado do
projeto (AUC-ROC 0,888), a apenas 0,05 do teto de 0,94 reportado pelos modelos vencedores
do desafio original (com deep learning). Bowel e spleen, apesar de CV razoável, mostram um
gap grande entre validação cruzada e teste — consistente com o tamanho pequeno das classes
raras, e discutido no artigo como limitação genuína, não falha do pipeline.

**Ablação:** combinar todas as famílias de descritores nem sempre é o melhor caminho — em
`kidney`, intensidade isolada é a pior opção entre as quatro (ver
`Outputs/Marco 3 - Experimentação/ablation_by_descriptor.csv`).

**Análise de erro (Marco 4):** as matrizes de confusão (`confusion_matrices.png`) revelam
vieses opostos entre órgãos — em `bowel`, o modelo nunca acerta a classe de lesão
(sensibilidade 0%); em `spleen`, o modelo quase nunca prevê "saudável" (sensibilidade da
classe saudável de só 8%). Detalhes por paciente em `predicoes_por_paciente.csv`.

## Uso de IA generativa

Declarado conforme exigido pelo enunciado (Seção 6): utilizado apoio de IA generativa
(Claude, Anthropic) para geração e revisão do código de EDA/pré-processamento/grade de
experimentos/análise de erro, depuração de bugs (desbalanceamento de classes em
RandomForest/XGBoost, viés de seleção de modelo pelo conjunto de teste, cálculo de
métricas AUC multiclasse) e para dúvidas de metodologia e redação do artigo final. Todo o
conteúdo foi revisado e validado pelo grupo.
