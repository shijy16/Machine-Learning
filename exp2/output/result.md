Bagging+DT, On Traning Set

| 最小叶子结点纯度 | 结果 |
| ---------------- | ---- |
| 0                |      |
| 0.01             |      |
| 0.05             |      |
| 0.08             |      |
| 0.09             |      |
| 0.1              |      |
| 0.11             |      |
| 0.12             |      |
| 0.15             |      |
| 0.2              |      |

Bagging+DT, On Test Set

最小叶子结点纯度

| 最小叶子结点纯度 | 结果    |
| ---------------- | ------- |
| 0.01             | 0.72335 |
| 0.1              | 0.72533 |

Bagging+SVM On Test Set

| 迭代次数  | 结果    |
| --------- | ------- |
| unlimited | 0.72567 |





由于训练集0,1标签数量差距大，AdaBoost训练轮数不能过多，以免在预测集上也出现偏置情况