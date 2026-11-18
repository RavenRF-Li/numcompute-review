# Assignment 2.2 最终提交清单

## 已验证

- [x] 本地工作分支为 `raven/assignment2.2`，`origin` 为 `https://github.com/RavenRF-Li/numcompute-review.git`。本地 `main` 与本地的 `origin/main` 跟踪记录指向原小组提交 `cec7f4a`。
- [x] 程序包含数据读取、增量预处理和统计、Gini 树、缓冲重训、Pipeline、累计与滚动指标、StreamTrainer 和三种绘图。包配置可构建含新旧两个包的 wheel。[运行入口](../README.md)
- [x] 最终全量回归为 709 项通过、0 项失败，含 642 项旧测试和 67 项 2.2 测试。4 条警告来自旧 benchmark 测试中的 NumPy 环境信息输出。[测试记录](test_plan.md)
- [x] 实际 `aiml` Jupyter kernel 已执行 5 个代码单元并保存三类图。已核对 Notebook、合成 CSV 和数据生成说明。[Notebook](../demo/stream_demo.ipynb) · [数据说明](../demo/data/README.md)
- [x] 候选 Gini 计分和 buffer 比较均有实测记录。记录中的 9 个源码 SHA-256 和 CSV SHA-256 与当前文件吻合。`git_base_commit` 记的是 D 提交前的 C 基线。[基准 JSON](../benchmark/stream_results.json)
- [x] D 阶段离线 wheel 已在临时 venv 验证仓库外导入、训练、预测和 Agg PNG。E 阶段对照确认程序、包配置、Notebook 和验证入口与 D 阶段一致，沿用已有验证。
- [x] 可编辑 Word 和由其导出的 3 页 PDF 已生成并逐页检查。标题、正文和 benchmark 表格可以在 Word 中编辑。[Word](../report/Assignment2.2-report.docx) · [PDF](../report/Assignment2.2-report.pdf)
- [x] 英文讲稿和操作表已准备，环境检查命令已执行验证。[讲稿](video_script.md) · [录制步骤](video_recording_steps.md)

## 待人工完成

- [ ] 审阅 Word，导出正式提交 PDF。任何编辑后重新导出，确认页数 **≤4**，并逐页检查分页、表格、图和文字。若将修改后的版本保留在仓库，显式暂存更新的报告文件并新增本地提交，再 push。
- [ ] 按录制步骤完成试录和正式录制，回放确认音画与总长度 **≤5 分钟**。视频仍待录制。
- [ ] 从仓库根目录 push 个人分支。以下命令供提交者执行。

```sh
git push -u origin raven/assignment2.2
```

- [ ] push 后打开个人分支页面，核对远端提交、文件和评分者访问权限。检查课程要求的 PDF 与视频提交形式。[个人分支](https://github.com/RavenRF-Li/numcompute-review/tree/raven/assignment2.2)

## 未完成的提交动作

- [ ] 将已核对的个人分支链接、正式 PDF 和视频文件或可访问链接提交至课程平台，再检查平台的提交确认。

本地材料已就绪，push 并核对远端后可提交链接。
