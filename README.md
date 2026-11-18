# NumCompute-Stream · Assignment 2.2

NumCompute-Stream 是一个模块化的流式分类框架。项目目标是从零实现一棵 NumPy 决策树，通过 `partial_fit` 接收数据批次，在缓冲达标时用整个缓冲区重训。

- `numcompute/` 是 Assignment 2.1 的基础包，保留数值工具和原有接口。
- `numcompute_stream/` 已实现数值 CSV 分批读取、Gini 树与缓冲重训、增量预处理、流式统计、Pipeline、分类指标、StreamTrainer 和可视化。2.2 的 Notebook 与基准实验均已执行。
- 2.2 的技术报告和视频留在阶段 E。`demo/quickstart.ipynb` 和 `benchmark/results.json` 保存 2.1 示例和测量结果。

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

以下流先用旧模型评估每批，再学习当前批。第二批同时带来数值 NaN 和新类别，第三批触发第二次重训。

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import DecisionTreeClassifier, Pipeline, StreamTrainer, TabularPreprocessor

pipe = Pipeline([
    ("prep", TabularPreprocessor([0], [1])),
    ("tree", DecisionTreeClassifier(buffer_size=2)),
])
trainer = StreamTrainer(pipe, labels=[0, 1], pos_label=1, window_size=2)
for X, y in [
    ([[0, 1], [2, 1]], [0, 1]),
    ([[float("nan"), 2]], [1]),
    ([[4, 2]], [0]),
]:
    row = trainer.fit_chunk(X, y)["log"]
    print(row["status"], row["retrain_count"], row["pending_samples"])
# warmup 1 0
# scored 1 1
# scored 2 0
PY
```

Pipeline 支持裸树、一个 `StandardScaler` 或 `TabularPreprocessor` 接树，以及前后逐行独立的保行无状态变换。含 NaN 的原始数据用组合器填补。Trainer 只将已经预测的样本纳入累计与滚动指标，`score_chunk` 只评分。日志保存标量摘要，`trainer.metrics.result()` 返回完整混淆矩阵和逐类结果。精确累计 AUC 保留历史分数并在查询时排序。日志分别记录预处理、tree fit、重训、整体更新和评估耗时，内存字段估算当前保留的 NumPy 数组存储。

```sh
# Assignment 2.2 当前全部测试
conda run -n aiml python -B -m unittest discover -s tests -p 'test_stream_*.py' -v

# Assignment 2.1 回归测试
conda run -n aiml python -B -m pytest -p no:cacheprovider tests --ignore-glob='tests/test_stream_*.py'

# 两套测试一起运行
conda run -n aiml python -B -m pytest -p no:cacheprovider tests
```

## Notebook、基准与安装验证

已执行的 [stream_demo.ipynb](demo/stream_demo.ipynb) 按 CSV 原顺序重放 2,000 条合成观测，展示先评估后学习、warmup、缓冲重训、滚动指标和分批统计。数据生成规则见 [CSV 数据说明](demo/data/README.md)。

在仓库根目录运行以下命令。Notebook 执行器使用现有 `aiml` Jupyter kernel 和本地 IPC，受限环境需要允许本地套接字。本机的 `aiml` 已提供 `jupyter_client`、`ipykernel` 和 `matplotlib_inline`，供执行 Notebook 使用。

```sh
conda run -n aiml python -B -m demo.execute_stream_notebook
conda run -n aiml python -B -m benchmark.run_stream
conda run -n aiml python -B -m demo.check_stream_install
```

- Notebook 保留实际输出，PNG 可单独查看。[累计准确率](demo/figures/stream_metric.png)、[共同批次模型比较](demo/figures/stream_compare.png)、[预测对照](demo/figures/stream_predictions.png)
- 基准命令会输出计时表和 JSON。循环与 NumPy 在计时前核对候选阈值和 Gini 增益。三个 buffer 使用共同可评估批次比较准确率，整体时间包含评分、精确 AUC 和内存日志。[实测 JSON](benchmark/stream_results.json)
- 安装检查离线构建 wheel，安装到临时 venv 并从仓库外导入两个包，运行训练与 Agg 绘图。临时 venv 复用 `aiml` 的 NumPy 和 matplotlib。

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
