# 12-block dynamic-scale comparison

- Images: 3
- Top-1 agreement: 0.00%
- Mean block11 output cosine: 0.077197
- Mean logit cosine: 0.082486

| image | PT | DLAU | match | block11 cosine | logit cosine |
|---|---:|---:|:---:|---:|---:|
| n01440764_10040.jpg | 0 | 986 | N | 0.122052 | 0.191357 |
| n01440764_10194.jpg | 0 | 220 | N | 0.079796 | 0.005635 |
| n01582220_10061.jpg | 18 | 979 | N | 0.029742 | 0.050466 |

## Attention heatmaps

- Directory: `/home/yingunix/AOC_final_project/DLA/DLA_model/results/heatmaps/12blocks`
- Maps are CLS-token attention over the 14x14 patch grid, averaged over 6 heads.
- Files are grouped by image and block: `block00` ... `block11`.
- Files per block: `pt_cls_attention.png`, `dlau_cls_attention.png`, `abs_diff_attention.png`, `pt_overlay.png`, `dlau_overlay.png`.

## Mean output error by block

| block | cosine | MAE | RMSE | Max abs error|
|---|---:|---:|---:|---:|
| 0 | 0.654251 | 0.950222 | 1.280831 | 11.863996 |
| 1 | 0.609271 | 0.978688 | 1.306639 | 11.242118 |
| 2 | 0.554930 | 1.055974 | 1.412004 | 11.602607 |
| 3 | 0.510937 | 1.113333 | 1.498283 | 12.736689 |
| 4 | 0.465884 | 1.262549 | 1.689267 | 14.213146 |
| 5 | 0.412603 | 1.461720 | 1.926173 | 36.044773 |
| 6 | 0.158984 | 2.414654 | 3.214994 | 216.924378 |
| 7 | 0.122099 | 4.135685 | 5.000120 | 241.013527 |
| 8 | 0.109992 | 5.651644 | 6.628903 | 188.392110 |
| 9 | 0.104400 | 7.631377 | 9.006505 | 216.771207 |
| 10 | 0.057483 | 10.730559 | 12.580303 | 242.309677 |
| 11 | 0.077197 | 13.715894 | 16.393356 | 282.707368 |
