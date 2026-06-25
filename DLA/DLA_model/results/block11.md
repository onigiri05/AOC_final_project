# Block11 dynamic-scale comparison

- Images: 3
- Top-1 agreement: 100.00%
- Mean block11 cosine: 0.818660
- Mean logit cosine: 0.778418

| image | PT | DLAU | match | block11 cosine | logit cosine |
|---|---:|---:|:---:|---:|---:|
| n01440764_10040.jpg | 0 | 0 | Y | 0.843648 | 0.813021 |
| n01440764_10194.jpg | 0 | 0 | Y | 0.828128 | 0.734843 |
| n01582220_10061.jpg | 18 | 18 | Y | 0.784206 | 0.787391 |

## Attention heatmaps

- Directory: `/home/yingunix/AOC_final_project/DLA/DLA_model/results/heatmaps/block11`
- Maps are CLS-token attention over the 14x14 patch grid, averaged over 6 heads.
- Files per image: `pt_cls_attention.png`, `dlau_cls_attention.png`, `abs_diff_attention.png`, `pt_overlay.png`, `dlau_overlay.png`.

## Mean error by stage

| stage | cosine | MAE | RMSE |
|---|---:|---:|---:|
| 00_input | 0.990979 | 0.505269 | 0.583384 |
| 01_norm1 | 0.987180 | 0.187916 | 0.225526 |
| 02_qkv | 0.989603 | 0.193271 | 0.252701 |
| 03_softmax | 0.911986 | 0.003435 | 0.017788 |
| 04_attention_value | 0.954344 | 0.289059 | 0.395500 |
| 05_projection | 0.793238 | 1.219029 | 1.439094 |
| 06_attention_residual | 0.937107 | 1.390443 | 1.707434 |
| 07_norm2 | 0.921994 | 0.387993 | 0.475949 |
| 08_fc1_linear | 0.971989 | 0.329212 | 0.425955 |
| 09_gelu | 0.239966 | 0.188992 | 0.428262 |
| 10_fc2 | 0.316767 | 1.590043 | 1.957595 |
| 11_block11_output | 0.818660 | 2.518359 | 3.000959 |
| 12_final_norm | 0.796603 | 1.204309 | 1.478672 |
| 13_classifier_logits | 0.778418 | 0.680275 | 0.860782 |
