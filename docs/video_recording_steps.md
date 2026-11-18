# 视频录制步骤

以 VS Code 的 Notebook 编辑器和集成终端录制。英文只读 [视频讲稿](video_script.md) 标注的正文。画面时间是建议，完整试读后以 5 分钟上限为准。

## 录制前打开窗口

1. 用 VS Code 打开 `numcompute-review` 仓库根目录。确认状态栏分支为 `raven/assignment2.2`，打开集成终端。终端工作目录为包含 `pyproject.toml`、`demo/` 和 `benchmark/` 的根目录。
2. 在编辑器打开 `demo/stream_demo.ipynb`、`demo/stream_workflow.py`、`numcompute_stream/stream.py` 和 `docs/implementation_plan.md`。为规划文件打开 Markdown 预览并定位到「可引用的实测结果」。该处两张表来自 `benchmark/stream_results.json`，录制时展示这些已保存数字。
3. 在 Notebook 右上角选择 **Python (AIML Course)**，kernel 名为 `aiml`。用下面命令在终端确认项目包和依赖可以导入。工作区已有执行过的 Notebook 和 PNG，可以先练习一次页面滚动。

```sh
conda run -n aiml python -B -c 'import sys, numpy, matplotlib, numcompute_stream; print("Python", sys.version.split()[0]); print("NumPy", numpy.__version__); print("Matplotlib", matplotlib.__version__); print("NumCompute-Stream import OK")'
```

预期当前机器输出 Python 3.12.13、NumPy 2.5.2、Matplotlib 3.11.1 以及 `NumCompute-Stream import OK`。

## 约三十秒的录前检查

用已有的屏幕录制工具选定 VS Code 窗口并选择麦克风。录约 30 秒，朗读讲稿第一段的前两句，然后从终端切到 Notebook 日志与 benchmark 表。回放试听音量、底噪和断音，确认终端整行、Notebook 输出、图例与表格数字都在画面里。合适后开始正式录制。可先在 Notebook 选择 Restart Kernel，保留已保存的仓库文件。

## 正式录制

| 讲稿 | 操作 | 等待或指向的内容 |
| --- | --- | --- |
| A · 0:00–0:35 | 在终端运行上方环境命令。切到 Notebook 顶部，按 **Shift+Enter** 执行第 1 个代码单元。 | 终端显示版本和导入成功。第一单元显示 `stream_sample.csv: nominal chunk size 32`。 |
| B · 0:35–1:15 | 打开 `demo/stream_workflow.py`，搜索 `def run_stream`。再到 `numcompute_stream/stream.py` 搜索 `self.pipeline.partial_fit(X, y)`。 | 先指向 `TabularPreprocessor`、`DecisionTreeClassifier`、`StreamTrainer` 的配置，再指向 `_score` 调用和随后的增量学习。 |
| C · 1:15–1:55 | 回到 Notebook，在 **Replay with a 64-row buffer** 下执行第 2 个代码单元。 | 日志前两批为 `warmup`，第 1 批 pending 32，第 2 批完成首次重训，第 3 批开始评分。底部显示 31 次重训和 16 行待训。 |
| D · 1:55–3:00 | 在 **Cumulative statistics and accuracy**、**Compare two buffer sizes**、**Pre-update predictions and ground truth** 下，依次执行第 3、4、5 个代码单元。 | 先指出统计和累计准确率图，再指出从第 9 批开始的两条比较曲线，最后显示 96 个样本的预测对照图。完整输出确认收集了 1,936 条学习前预测。 |
| E · 3:00–4:20 | 切到规划文档的「可引用的实测结果」，先展示 Gini kernel 表，再展示 buffer 表。 | 画面显示保存的实测数字和 1,744 条共同样本。现场每批耗时会变，讲稿中的具体数值以保存的实验表为准。结语读完后停止录制。 |

如录制前需要在终端确认整份 Notebook 可执行，可在根目录运行以下命令。它会使用 `aiml` 的真实 Jupyter kernel 并重新保存 Notebook 输出和 PNG。若保留这次重跑的文件供提交，检查差异后再提交。

```sh
conda run -n aiml python -B -m demo.execute_stream_notebook
```

## 保存与交付前回放

用现有录制工具结束并保存视频。若工具提供 MP4 导出，可保存为 `Assignment2.2-demo.mp4`。回放确认总长度 **≤5 分钟**，开头清楚说明环境与数据，所有图表和 benchmark 数字可读，结尾完整。按课程允许的方式提交视频文件或链接。视频录制和访问检查保留在 [最终提交清单](submission_checklist.md) 中。
