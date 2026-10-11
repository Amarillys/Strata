# GSQ IQ3_XXS 与 v4 IQ3_KT 的逐层专家分配

层号从 0 开始。每个 gate/up 类型同时用于该层两个张量，每个张量覆盖 512 个专家。
v2 的全部 gate、up、down 均为 IQ3_KT。此表由 analyze.py 读取本地 GGUF 头部生成。

| 层 | GSQ gate/up | GSQ down | v4 gate/up | v4 down |
| ---: | --- | --- | --- | --- |
| 0 | IQ2_XS | IQ4_NL | IQ4_KT | IQ3_KT |
| 1 | IQ2_XXS | Q2_0 | IQ4_KT | IQ3_KT |
| 2 | IQ2_XS | IQ4_NL | IQ4_KT | IQ3_KT |
| 3 | IQ2_XXS | Q2_0 | IQ4_KT | IQ3_KT |
| 4 | IQ2_XS | IQ4_NL | IQ4_KT | IQ3_KT |
| 5 | IQ2_XS | Q2_0 | IQ4_KT | IQ3_KT |
| 6 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 7 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 8 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 9 | IQ2_XS | Q2_0 | IQ3_KT | IQ3_KT |
| 10 | IQ2_XS | Q2_0 | IQ3_KT | IQ3_KT |
| 11 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 12 | IQ2_S | Q2_0 | IQ3_KT | IQ3_KT |
| 13 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 14 | IQ2_XS | Q2_0 | IQ3_KT | IQ3_KT |
| 15 | IQ2_S | Q2_0 | IQ3_KT | IQ3_KT |
| 16 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 17 | IQ2_S | Q2_0 | IQ3_KT | IQ3_KT |
| 18 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 19 | IQ2_XS | Q2_0 | IQ3_KT | IQ3_KT |
| 20 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 21 | IQ2_S | Q2_0 | IQ3_KT | IQ3_KT |
| 22 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 23 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 24 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 25 | IQ2_XS | Q2_0 | IQ3_KT | IQ3_KT |
| 26 | IQ2_S | Q2_0 | IQ3_KT | IQ3_KT |
| 27 | IQ2_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 28 | IQ3_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 29 | IQ2_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 30 | IQ3_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 31 | IQ2_XXS | IQ4_NL | IQ3_KT | IQ3_KT |
| 32 | IQ3_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 33 | IQ3_S | Q2_0 | IQ3_KT | IQ3_KT |
| 34 | IQ3_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 35 | IQ3_XXS | IQ4_NL | IQ3_KT | IQ3_KT |
| 36 | IQ3_XXS | Q2_0 | IQ3_KT | IQ3_KT |
| 37 | IQ3_S | IQ4_NL | IQ3_KT | IQ3_KT |
| 38 | IQ2_S | IQ4_NL | IQ4_KT | IQ3_KT |
| 39 | IQ2_XS | IQ4_NL | IQ4_KT | IQ3_KT |
| 40 | IQ2_S | Q2_0 | IQ4_KT | IQ3_KT |
| 41 | IQ3_XXS | Q2_0 | IQ4_KT | IQ3_KT |
| 42 | IQ2_S | IQ4_NL | IQ4_KT | IQ3_KT |
| 43 | IQ3_S | IQ4_NL | IQ4_KT | IQ3_KT |
| 44 | IQ3_XXS | IQ4_NL | IQ4_KT | IQ3_KT |
| 45 | IQ3_XXS | Q2_0 | IQ4_KT | IQ3_KT |
| 46 | IQ2_S | IQ4_NL | IQ4_KT | IQ3_KT |
| 47 | IQ3_XXS | IQ4_NL | IQ4_KT | IQ3_KT |
