# NumCompute-Stream · Assignment 2.2

NumCompute-Stream 是一个模块化的流式分类框架。项目目标是从零实现一棵 NumPy 决策树，通过 `partial_fit` 接收数据批次，在缓冲达标时用整个缓冲区重训。

- `numcompute/` 是 Assignment 2.1 的基础包，保留数值工具和原有接口。
- `numcompute_stream/` 是 Assignment 2.2 的扩展包。已实现数值 CSV 分批读取、输入验证、Gini 分类树、缓冲重训、增量预处理和流式统计。
- 流式指标、Pipeline、StreamTrainer、绘图，以及 2.2 的 demo 和 benchmark 仍待实现。现有 `demo/` 和 `benchmark/results.json` 保存 2.1 示例和测量结果。

## 运行与测试

在仓库根目录，使用已有的 Conda `aiml` 环境运行。以下示例读取样例 CSV 的第一批，以最后一列为标签。两组数组均为 float64，特征缺失值保留为 NaN。

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import iter_supervised_csv

batches = iter_supervised_csv(
    "demo/data/sample.csv", target_col=-1, chunk_size=2, skip_header=1
)
try:
    X, y = next(batches)
    print(X.shape, y.tolist())  # (2, 2) [0.0, 1.0]
finally:
    batches.close()
PY
```

以下示例在缓冲区累积到四行后训练。预测概率按 `classes_` 排列，独立树要求有限的数值特征。

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import DecisionTreeClassifier

tree = DecisionTreeClassifier(buffer_size=4, max_depth=1)
tree.partial_fit([[0], [1]], ["low", "low"])
tree.partial_fit([[2], [3]], ["high", "high"])
print(tree.predict([[0], [3]]).tolist())  # ['low', 'high']
print(tree.classes_.tolist(), tree.predict_proba([[3]]).tolist())  # ['high', 'low'] [[1.0, 0.0]]
PY
```

预处理快照固定了数值变换和输出列。下面用两批数据更新混合列预处理器，并对同一行比较快照与当前变换。

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import TabularPreprocessor

prep = TabularPreprocessor(numeric_columns=[0], categorical_columns=[1])
prep.partial_fit([[0, 1], [2, 1]])
frozen = prep.snapshot()
prep.partial_fit([[4, 2]])
print(frozen.transform([[2, 2]]).tolist())  # [[1.0, 0.0]]
print(prep.transform([[2, 2]]).tolist())    # [[0.0, 0.0, 1.0]]
PY
```

`StreamingStats(bin_edges=...)` 按列累计均值、方差和固定箱直方图，通过 `result(q=0.5)` 查询精确分位数。精确分位数保留历史样本，内存随累计行数增长。

```sh
# Assignment 2.2 当前全部测试
conda run -n aiml python -B -m unittest discover -s tests -p 'test_stream_*.py' -v

# Assignment 2.1 回归测试
conda run -n aiml python -B -m pytest -p no:cacheprovider tests --ignore-glob='tests/test_stream_*.py'

# 两套测试一起运行
conda run -n aiml python -B -m pytest -p no:cacheprovider tests
```

## 规划与归档

- [实施计划](docs/implementation_plan.md)和[测试计划](docs/test_plan.md)记录 2.2 的结构、进度与验收场景。
- [旧 API 契约](API_CONTRACT.md)适用于 `numcompute/`。
- [Assignment 2.1 归档](archive/assignment2.1/)保留[原 README](archive/assignment2.1/README.md)、作业要求、分工、贡献记录和历史报告（[PDF](archive/assignment2.1/report/NumCompute-report.pdf)、[DOCX](archive/assignment2.1/report/NumCompute-report.docx)）。归档文档里的运行命令及代码、测试、demo 和 benchmark 路径以仓库根目录为基准。

## 最终交付清单

- 完整的 `numcompute_stream/`、README 和至少 30 项有意义的 2.2 单元测试。
- 可执行的 `demo/stream_demo.ipynb`，演示 CSV 分批读取、增量训练、指标和绘图。
- `benchmark/` 中可复现的循环与向量化比较，以及不同 `buffer_size` 的实测结果。
- 最多 4 页的 PDF 技术报告和最多 5 分钟的演示视频。
- 在个人 fork 的 `raven/assignment2.2` 分支开发，`main` 保留原始小组代码。
