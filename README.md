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
do enunciado.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── fichamento.md                          # resumo dos 10 artigos lidos
├── formulacao_problema.md                 # definição da tarefa (entrada/saída/unidade de amostra)
├── notebooks/
│   ├── marco-1-eda-abdominal-trauma-g7.ipynb
│   ├── marco-2-pre-processing-abdominal-trauma-g7.ipynb
│   └── marco-3-g7-abdominal-trauma.ipynb   # extração de descritores + grade de experimentos
└── outputs/
    ├── class_distribution_by_organ.png
    ├── class_distribution_summary.csv
    ├── example_windowing.png
    ├── examples_by_class.png
    ├── preprocessing_before_after.png
    ├── sample_patient_ids.csv
    ├── train_test_split.csv
    ├── intensity_features.csv
    ├── texture_shape_features.csv
    ├── labeled_features_full.csv
    ├── baseline_trivial_results.csv
    ├── first_model_results.csv             # Regressão Logística (Semana 2)
    ├── grid_experiments_results.csv        # grade completa: 5 órgãos × 4 descritores × 3 modelos
    ├── summary_table_descritor_modelo.csv
    ├── ablation_by_descriptor.csv
    ├── ablation_chart.png
    ├── best_combination_per_organ.csv
    ├── comparison_baseline_vs_best.csv
    └── feature_importance_top15.png
    └── feature_importance_top15 Liver.png
```

## Dados

Os dados **não estão neste repositório** (política do próprio Kaggle e do enunciado do
TP1 — dados brutos não vão para o repositório).

- Fonte: [RSNA 2023 Abdominal Trauma Detection — Kaggle](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- É necessário aceitar os termos da competição no Kaggle antes de acessar os dados.
- A amostra estratificada usada no baseline (lista de `patient_id`, estratificada por
  `any_injury`, semente fixa `SEED=42`) está documentada em `outputs/sample_patient_ids.csv`.
- **Tamanho da amostra: 500 pacientes** (ampliada a partir dos 180 pacientes usados nas
  primeiras versões do Marco 2/3 — o teste com 36 pacientes se mostrou instável demais para
  algumas classes raras; com 500 pacientes o split fica em 400 treino / 100 teste).
- A partição treino/teste (`outputs/train_test_split.csv`) é **congelada** desde a
  reamostragem — não deve ser recalculada nas próximas semanas.

## Como rodar

1. Acesse o Kaggle e crie um **Notebook** vinculado à competição
   `rsna-2023-abdominal-trauma-detection` (o dataset já vem montado automaticamente em
   `/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection`).
2. Rode em sequência:
   - `marco-1-eda-abdominal-trauma-g7.ipynb` (EDA + amostragem estratificada);
   - `marco-2-pre-processing-abdominal-trauma-g7.ipynb` (pipeline de pré-processamento,
     baseline trivial, primeiro modelo — Regressão Logística);
   - `marco-3-g7-abdominal-trauma.ipynb` (extração das 3 famílias de descritores + grade de
     experimentos descritor × modelo, com validação cruzada e ajuste de hiperparâmetros).
3. Confirme que a variável `DATA_DIR`, na célula de configuração de cada notebook, aponta para:
   ```python
   DATA_DIR = Path("/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection")
   ```
4. **Se cada marco estiver rodando em uma sessão separada do Kaggle** (não é a mesma sessão
   contínua desde o Marco 1): anexe o notebook do marco anterior como **Input** do notebook
   atual (painel direito → "+ Add Input" → aba "Notebooks" → selecionar a versão salva do
   marco anterior). Isso é necessário porque os artefatos "congelados" de um marco
   (`sample_patient_ids.csv`, `train_test_split.csv`, `intensity_features.csv`,
   `texture_shape_features.csv`, `labeled_features_full.csv` etc.) precisam estar acessíveis
   para o marco seguinte reaproveitá-los em vez de recalculá-los do zero. Os notebooks já têm
   uma célula que procura esses arquivos tanto na pasta de trabalho local quanto dentro de
   `/kaggle/input/` (onde ficam os notebooks anexados como Input), então não é necessário
   descobrir o caminho exato manualmente — só anexar o Input correto antes de rodar.
5. Rode as células em sequência (Run All). As saídas (gráficos e CSVs) são geradas
   automaticamente na pasta de trabalho do Kaggle e podem ser baixadas em seguida.

> **Nota sobre tempo de execução:** a extração de descritores e a grade de experimentos
> (Marco 3) demoram bem mais que os notebooks anteriores — a grade completa (5 órgãos × 4
> descritores × 3 modelos, com `GridSearchCV`) leva cerca de 1 hora com 500 pacientes. Se
> for só reaproveitar um resultado já salvo (ex.: regenerar uma figura), não é necessário
> rodar a grade inteira de novo — os CSVs de saída (`grid_experiments_results.csv`,
> `best_combination_per_organ.csv`) já bastam para refazer tabelas e gráficos.

## Reprodutibilidade

- Semente fixa: `SEED = 42` (usada na amostragem estratificada, na partição treino/teste,
  na validação cruzada e em qualquer etapa aleatória).
- Nenhum caminho absoluto pessoal é usado — apenas o caminho padrão de input do Kaggle.
- Desbalanceamento de classes tratado explicitamente: `class_weight="balanced"` (SVM),
  `class_weight="balanced_subsample"` + `sample_weight` (Random Forest) e `sample_weight`
  via `compute_sample_weight("balanced", ...)` (XGBoost, que não tem parâmetro
  `class_weight` nativo).
- Dependências declaradas em `requirements.txt`.

## Status do projeto

- [x] Marco 1 — EDA, leitura DICOM, amostragem, fichamento, formulação do problema
- [x] Marco 2 — pipeline de pré-processamento completo, partição por paciente, baseline
      trivial, primeiro modelo (Regressão Logística)
- [x] Marco 3 — extração das 3 famílias de descritores (intensidade, textura GLCM, forma),
      grade de experimentos descritor × modelo (SVM, Random Forest, XGBoost) com validação
      cruzada, estudo de ablação por família de descritor, amostra ampliada para 500
      pacientes
- [ ] Marco 4 — análise de erro, escrita do artigo, teste de reprodutibilidade

### Principais resultados (Marco 3, teste com 100 pacientes)

| Órgão | Baseline trivial | Melhor combinação | Balanced Accuracy | AUC-ROC |
|---|---|---|---|---|
| bowel | 0,500 | intensidade + SVM | 0,724 | 0,668 |
| extravasation | 0,500 | forma + XGBoost | 0,589 | 0,550 |
| kidney | 0,333 | combinado + SVM | 0,780 | 0,888 |
| liver | 0,333 | intensidade + RandomForest | 0,456 | 0,565 |
| spleen | 0,333 | textura + SVM | 0,437 | 0,609 |

Todos os 5 órgãos superam o baseline trivial. Achado de ablação relevante: combinar todas
as famílias de descritores nem sempre é o melhor caminho — em `kidney` e `liver`, uma
família isolada supera o conjunto combinado, sinal de alta dimensionalidade relativa ao
tamanho da amostra (ver `formulacao_problema.md` e a seção de ablação nos outputs).

## Uso de IA generativa

Declarado conforme exigido pelo enunciado (Seção 6): utilizado apoio de IA generativa
(Claude, Anthropic) para geração e revisão do código de EDA/pré-processamento/grade de
experimentos, depuração de bugs (desbalanceamento de classes em RandomForest/XGBoost,
cálculo de métricas AUC multiclasse) e para dúvidas de metodologia. Todo o conteúdo foi
revisado e validado pelo grupo.
