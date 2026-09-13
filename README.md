# TP1 — Baseline de Projeto de Pesquisa (G7)

**Disciplina:** Tópicos Especiais em Sistemas de Informação
**Professor:** Décio Gonçalves de Aguiar Neto
**Grupo G7:** Luis Antônio Pereira Rabelo · Francisco Ruan da Silva Queiroz · Carlos Daniel da Silva Ribeiro
**Desafio RSNA atribuído:** [2023 — Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

## Objetivo

Construir um baseline com **métodos clássicos de visão computacional e aprendizado de
máquina** (sem redes neurais profundas) para o problema de detecção e classificação de
lesões traumáticas abdominais em exames de TC, usando o dataset do desafio RSNA 2023.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── fichamento.md              # resumo dos 5 artigos lidos
├── formulacao_problema.md     # definição da tarefa (entrada/saída/unidade de amostra)
├── notebooks/
│   └── eda_abdominal_trauma_G7.ipynb
└── outputs/
    ├── class_distribution_by_organ.png
    ├── class_distribution_summary.csv
    ├── example_windowing.png
    ├── examples_by_class.png
    └── sample_patient_ids.csv
```

## Dados

Os dados **não estão neste repositório** (política do próprio Kaggle e do enunciado do
TP1 — dados brutos não vão para o repositório).

- Fonte: [RSNA 2023 Abdominal Trauma Detection — Kaggle](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- É necessário aceitar os termos da competição no Kaggle antes de acessar os dados.
- A amostra estratificada usada no baseline (lista de `patient_id`, com semente fixa
  `SEED=42`) está documentada em `outputs/sample_patient_ids.csv`.

## Como rodar

1. Acesse o Kaggle e crie um **Notebook** vinculado à competição
   `rsna-2023-abdominal-trauma-detection` (o dataset já vem montado automaticamente em
   `/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection`).
2. Faça upload do notebook `notebooks/eda_abdominal_trauma_G7.ipynb` para esse ambiente
   (ou copie o conteúdo das células).
3. Confirme que a variável `DATA_DIR`, na célula de configuração, aponta para:
   ```python
   DATA_DIR = Path("/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection")
   ```
4. Rode as células em sequência (Run All). As saídas (gráficos e CSVs) são geradas
   automaticamente na pasta de trabalho do Kaggle e podem ser baixadas em seguida.

## Reprodutibilidade

- Semente fixa: `SEED = 42` (usada na amostragem estratificada e em qualquer etapa
  aleatória).
- Nenhum caminho absoluto pessoal é usado — apenas o caminho padrão de input do Kaggle.
- Dependências declaradas em `requirements.txt`.

## Status do projeto

- [x] Marco 1 — EDA, leitura DICOM, amostragem, fichamento, formulação do problema
- [ ] Marco 2 — pipeline de pré-processamento completo, partição por paciente, baseline trivial
- [ ] Marco 3 — extração de características (≥3 famílias), grade de experimentos descritor × modelo
- [ ] Marco 4 — análise de erro, escrita do artigo, teste de reprodutibilidade

## Uso de IA generativa

Declarado conforme exigido pelo enunciado (Seção 6): utilizado apoio de IA generativa
(Claude, Anthropic) para geração e revisão do código de EDA/pré-processamento e para
dúvidas de metodologia. Todo o conteúdo foi revisado e validado pelo grupo.
