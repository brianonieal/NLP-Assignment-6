| run | model | where | batch | lr | epochs | final dev | final test | tuned params | minutes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3.1.1_overfit_distilbert | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-4 | 20 | 1.0000 | 1.0000 |  |  |
| 3.1.2_full30_distilbert | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-4 | 30 | 0.7021 | 0.6931 |  | 30.2 |
| 3.1.3_grid_distilbert_lr0.0001_ep7 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-4 | 7 | 0.7055 | 0.7050 |  | 7.3 |
| 3.1.3_grid_distilbert_lr0.0001_ep9 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-4 | 9 | 0.7141 | 0.7057 |  | 9.2 |
| 3.1.3_grid_distilbert_lr0.0005_ep7 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 5e-4 | 7 | 0.6217 | 0.6230 |  | 7.5 |
| 3.1.3_grid_distilbert_lr0.0005_ep9 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 5e-4 | 9 | 0.6217 | 0.6230 |  | 9.5 |
| 3.1.3_grid_distilbert_lr0.001_ep7 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-3 | 7 | 0.6217 | 0.6230 |  | 7.5 |
| 3.1.3_grid_distilbert_lr0.001_ep9 | distilbert-base-uncased | laptop RTX 4050, fp32 | 64 | 1e-3 | 9 | 0.6217 | 0.6230 |  | 9.7 |
| 3.2.1_roberta_full | roberta-base | laptop RTX 4050, fp32 | 32 | 2e-5 | 6 | 0.7856 | 0.7730 | 124745474 | 17.3 |
| 3.2.2_roberta_head | roberta-base | laptop RTX 4050, fp32 | 32 | 1e-3 | 10 | 0.6425 | 0.6391 | 1538 | 17.2 |
| 3.2.3_roberta_prefix | roberta-base | laptop RTX 4050, bf16 | 16 | 5e-3 | 10 | 0.6550 | 0.6580 | 99842 | 47.5 |
| 3.2.6_roberta_lora | roberta-base | laptop RTX 4050, bf16 | 32 | 5e-4 | 6 | 0.7618 | 0.7540 | 887042 | 13.2 |
| 3.1.4_check_distilbert_amp_lr0.0001_ep9 | distilbert-base-uncased | laptop RTX 4050, bf16 | 64 | 1e-4 | 9 | 0.7208 | 0.7085 |  | 6.7 |
| 3.1.4_control_roberta-base_fp32_lr0.0001_ep7 | roberta-base | Colab L4, fp32 | 32 | 1e-4 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_extra_roberta-base_lr2e-05_ep9 | roberta-base | Colab L4, bf16 | 32 | 2e-5 | 9 | 0.7636 | 0.7477 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.0001_ep7 | bert-base-uncased | Colab L4, bf16 | 32 | 1e-4 | 7 | 0.6869 | 0.6734 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.0001_ep9 | bert-base-uncased | Colab L4, bf16 | 32 | 1e-4 | 9 | 0.7055 | 0.6734 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.0005_ep7 | bert-base-uncased | Colab L4, bf16 | 32 | 5e-4 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.0005_ep9 | bert-base-uncased | Colab L4, bf16 | 32 | 5e-4 | 9 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.001_ep7 | bert-base-uncased | Colab L4, bf16 | 32 | 1e-3 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_bert-base-uncased_lr0.001_ep9 | bert-base-uncased | Colab L4, bf16 | 32 | 1e-3 | 9 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.0001_ep7 | roberta-base | Colab L4, bf16 | 32 | 1e-4 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.0001_ep9 | roberta-base | Colab L4, bf16 | 32 | 1e-4 | 9 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.0005_ep7 | roberta-base | Colab L4, bf16 | 32 | 5e-4 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.0005_ep9 | roberta-base | Colab L4, bf16 | 32 | 5e-4 | 9 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.001_ep7 | roberta-base | Colab L4, bf16 | 32 | 1e-3 | 7 | 0.6217 | 0.6230 |  |  |
| 3.1.4_grid_roberta-base_lr0.001_ep9 | roberta-base | Colab L4, bf16 | 32 | 1e-3 | 9 | 0.6217 | 0.6230 |  |  |
