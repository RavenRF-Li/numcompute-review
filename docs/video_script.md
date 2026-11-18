# NumCompute Stream 英文演示讲稿

英文正文可直接朗读。每段的时间和画面提示供录制时看，具体切换见 [录制步骤](video_recording_steps.md)。

英文正文共 **480 词**。按每分钟 130 词估算，朗读约 **3 分 42 秒**。含切屏和等待，建议按 **4 分 20 秒** 排练，以实际试读为准。

## A 环境与数据

**0:00–0:35 · 画面**　VS Code 终端显示环境检查。转到 Notebook 顶部，运行第一段代码。

**英文正文**

Hi, I’m Ruofan Li. This is NumCompute Stream, my Assignment Two Point Two project. It uses Python, NumPy, and Matplotlib. I’m running the notebook with the AIML Course kernel. The CSV contains two thousand synthetic rows. Each chunk has up to thirty two samples. The stream includes missing numeric values, new categories, and a change in the label rule halfway through.

## B Pipeline 配置和更新关系

**0:35–1:15 · 画面**　打开 `demo/stream_workflow.py` 的 `run_stream`，再打开 `numcompute_stream/stream.py` 的 `fit_chunk`。

**英文正文**

The workflow builds a pipeline with a tabular preprocessor and a decision tree. The preprocessor fills missing numeric values, scales the numeric columns, and encodes categories. The trainer scores a chunk first, then calls the pipeline’s partial fit method. When the raw buffer reaches its limit, the pipeline transforms all buffered rows with one preprocessing snapshot and fits a new tree. It publishes the snapshot and the tree together. That pair serves predictions until the next build.

## C Warmup 缓冲和日志

**1:15–1:55 · 画面**　回 Notebook，运行 `Replay with a 64-row buffer` 下的代码单元，指向前 4 行日志和最后汇总。

**英文正文**

Here the buffer holds sixty four rows. The first chunk leaves thirty two rows pending. The second chunk fills the buffer and triggers the first build. Both are warmup chunks, so scoring starts at chunk three. Each log row shows the model used for scoring, the build count, pending rows, and timing. The memory column estimates retained NumPy arrays. This run made thirty one builds and ended with sixteen rows pending.

## D 统计 预测和三类图

**1:55–3:00 · 画面**　依次运行统计和准确率、模型比较、预测对照三个代码单元。

**英文正文**

The statistics section shows observed counts, running means and variances, exact quartiles, and fixed bin counts. The first plot shows cumulative accuracy. It begins with a gap for warmup. The sixty four row run scored one thousand nine hundred and thirty six predictions, with about seventy eight percent accuracy. The log also reports a rolling window of the latest one hundred and twenty eight evaluated rows.

The second plot compares actual runs with buffers of sixty four and two hundred and fifty six. Both curves start at chunk nine, when both models can score. The third plot shows the first ninety six predictions beside their true labels. A cross over a circle marks a correct prediction. Each prediction was made before learning from that sample’s chunk.

## E 实测 benchmark 与结语

**3:00–4:20 · 画面**　切到 `docs/implementation_plan.md` Markdown 预览的「可引用的实测结果」，依次指向两张表和共同样本说明。

**英文正文**

These tables show the saved benchmark results. For one feature, vectorised Gini candidate scoring ran about seven point six to seven point nine times as fast as the Python loop. The trial checked split thresholds and gains before timing.

The buffer comparison uses the same one thousand seven hundred and forty four samples for all three models. Buffers of sixty four, one hundred and twenty eight, and two hundred and fifty six scored seventy seven point eight one, seventy nine point three six, and seventy six point four three percent. The middle buffer had the highest accuracy on this stream. The largest buffer scored fewer rows overall and finished the replay fastest. The timing columns show totals for the full stream. Overall time includes scoring, AUC queries, and logging.

The repository includes the notebook, the data generator, and the benchmark results. Thank you for watching.
