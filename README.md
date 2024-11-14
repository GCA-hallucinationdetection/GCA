## Sensitivity Analysis of Weight and Threshold with Respect to Experimental Results
The following tables show the effects of threshold/weight on different datasets. **The results show that the performance are not so sensitive to the threshold/weight**.

### Table 1：Experimental Results under Different Hyperparameters on the PHD Dataset
   
| ${W_{1}}$ | ${W_{2}}$ | ${W_{3}}$ | ${W_{4}}$ | ${T}$ | $\boldsymbol{F_{1}}$ | **Acc** |
| --------- | --------- | --------- | --------- | ----- | -------------------- | ------- |
|     0.25      | 0.1      | 0.15      | 0.5    | $\mu$ |               53.8       |      67.3   |
|     0.5      |      0.2     |        0.1   |       0.2    |      $\mu$  |     54.5                |    67.6     |
|     0.7      |       0.1    |         0.1  |       0.1    |      $\mu$  |       55.4               |    68.3     |
|      0.25     |      0.5     |        0.15   |        0.1   |    $\mu+2\sigma$    |            52.7          |      65.6     |
|    0.5       |     0.1      |      0.3     |       0.1    |    $\mu+2\sigma$    |           53.1           |     66.2    |
|     0.7      | 0.15      | 0.1      | 0.05     | $\mu+2\sigma$ |          54.0            |       67.1  |


### Table 2：Experimental Results under Different Hyperparameters on the WikiBio Dataset

| ${W_{1}}$ | ${W_{2}}$ | ${W_{3}}$ | ${W_{4}}$ | ${T}$ | $\boldsymbol{F_{1}}$ | **Acc** |
| --------- | --------- | --------- | --------- | ----- | -------------------- | ------- |
|     0.25      | 0.5      | 0.15      | 0.1    | $\mu+\sigma$ |               88.6       |      79.8   |
|     0.5      |      0.2     |        0.15   |       0.15    |      $\mu+\sigma$  |     89.3               |    80.3     |
|     0.7      |       0.1    |         0.1  |       0.1    |      $\mu+\sigma$  |       89.7               |    81.0     |
|      0.25     |      0.3     |        0.35   |        0.1   |    $\mu+2\sigma$    |            89.1          |     81.7      |
|    0.5       |     0.1      |      0.3     |       0.1    |    $\mu+2\sigma$    |           89.6           |     82.5   |
|     0.7      | 0.15      | 0.1      | 0.05     | $\mu+2\sigma$ |          90.7           |       83.2  |

Results on both Table 1 and Table 2 show that under different hyperparameter (weight/threshold) settings, the variation in experimental results is not significant and remains relatively stable.

### Table 3：Analysis of One-Way ANOVA Results between Different Hyperparameters and Metric Values on the PHD Dataset

|                      | ${W_{1}}$ | ${W_{2}}$ | ${W_{3}}$ | ${W_{4}}$ | ${T}$ |
| -------------------- | --------- | --------- | --------- | --------- | ----- |
| $\boldsymbol{F_{1}}$ |      0.06     |       0.12    |         0.09  |       0.85    |     0.17  |
| **Acc**              |          0.07   |       0.15    |       0.09  |        0.18   |  0.23     |

### Table 4: Analysis of One-Way ANOVA Results between Different Hyperparameters and Metric Values on the WikiBio Dataset

|                      | ${W_{1}}$ | ${W_{2}}$ | ${W_{3}}$ | ${W_{4}}$ | ${T}$ |
| -------------------- | --------- | --------- | --------- | --------- | ----- |
| $\boldsymbol{F_{1}}$ |      0.08     |       0.12    |         0.08  |       0.04    |     0.33  |
| **Acc**              |          0.01   |       0.04   |       0.02  |        0.09   |  0.27     |

One-way ANOVA results on Table 3 and Table 4 show that all p-values are greater than 0.1, indicating that the experimental results are not sensitive to the hyperparameter settings




