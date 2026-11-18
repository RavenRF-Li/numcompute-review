# NumCompute-Stream 实施计划

## 1. 仓库检查与测试基线

检查时工作分支为 `raven/assignment2.2`，HEAD 和本地 `main` 均指向 `cec7f4ad55cddc738e5b5cb2c3212b3ac4fc0033`。`origin` 指向个人账户下的仓库，GitHub fork 的上游关联尚未核验。仓库及其上级目录没有找到适用的 `AGENTS.md`。已有未跟踪文件 `.vscode/settings.json` 已保留。

阅读了 `pyproject.toml`、[旧 README](../archive/assignment2.1/README.md)、[旧 API 契约](../API_CONTRACT.md)、[旧作业要求](../archive/assignment2.1/ASSIGNMENT_REQUIREMENTS.md)、[旧小组分工](../archive/assignment2.1/GROUP_ALLOCATION.md)，以及下表涉及的模块、测试和 demo 入口。旧文档的分工、20 项测试门槛以及无需实现模型的约定属于 2.1。本计划以本次提供的 2.2 正文和 specs 为准。

在仓库根目录用已有 Conda `aiml` 环境运行全量测试。默认 Homebrew Python 3.14.7 未找到 pytest，现有 `aiml` 无需安装依赖即可执行。

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/aiml/bin/python -B -m pytest -p no:cacheprovider tests
```

实际环境为 macOS 27.2 arm64、Python 3.12.13、NumPy 2.5.2、pytest 9.2.0。8 个测试文件共收集 642 项，**642 passed，0 failed，4 warnings，0.80 s，退出码 0**。四条警告均来自 benchmark 测试调用 `np.show_config()` 时 NumPy 建议安装 PyYAML 以改善输出。pytest 用于执行现有测试，2.2 新测试使用标准库 `unittest`。这是规划阶段记录的旧基线，第 1 阶段复跑结果见下文。

### 复用、改造与新增

| 范围 | 实际代码及行为 | 2.2 处理 |
| --- | --- | --- |
| CSV I/O | `numcompute/io.py` 的 `load_csv` 用 `np.genfromtxt` 一次读入数值数据，处理 header、delimiter、dtype 和缺失值。`iter_csv_chunks` 用 `islice` 分批调用读取逻辑，检查跨批列数，跳过空批。`tests/test_io_preprocessing.py` 覆盖尾批、缺失、列数变化。 | **直接复用**两个读取器，在新 `io.py` 重新导出。**新增**带标签列的惰性包装，保留特征 NaN，检查标签和列索引。初版使用数值 CSV，类别用数值编码。 |
| 数值预处理 | `numcompute/preprocessing.py` 的 `StandardScaler.fit` 用 `nanmean/nanstd` 重算整批，常量列使用 scale 1。`SimpleImputer.fit` 支持 mean、median、constant。两者的统计拟合均拒绝全 NaN 列。`MinMaxScaler` 也是批量拟合。 | **改造后复用**标准化及 `np.where` 填补公式，原有类保留。新类用 `RunningStats` 累计，均值填补支持全缺失列的 fallback。`MinMaxScaler` 及 median 填补留在旧包。 |
| 编码 | 同文件 `OneHotEncoder.fit` 对 `(feature, category)` 去重排序，按原列分组；`transform` 使用广播匹配和 `logical_or.reduceat` 检查未知类别。再次拟合可能移动既有列。`tests/test_io_preprocessing.py` 重复声明同名 `TestOneHotEncoder`，Python 只保留后一次定义供收集。 | **改造后复用**向量化匹配思路，**新增**全局只追加的列映射，未知检查按原列聚合。把跨批列稳定性纳入新测试。 |
| 统计 | `numcompute/stats.py` 的 `RunningStats.update/merge/copy` 已支持分批矩估计和 NaN 策略。现有 `histogram` 展平后单次计数，`quantile` 对传入数组单次归约。 `tests/test_stats_metrics.py` 对比不等长分批与批量矩，并覆盖 merge；`tests/test_stats_metrics_edges.py` 有方差精度边界。 | **直接复用** `RunningStats` 的累计计算与 `quantile` 的精确归约。**新增** `update_stats`、按列调用同一 NumPy 直方图 kernel 及分位数样本管理。固定边缘在构造时统一验证。 |
| 指标 | `numcompute/metrics.py` 中 `confusion_matrix` 支持显式标签顺序，内部用 `bincount`；accuracy、precision、recall、F1 均从输入批计算。`roc_auc_score` 是精确二分类，合并同分，单一真实类别会报错。现有测试覆盖顺序、平局和未定义的分母。 | **复用**二分类 AUC 和 `bincount` 计数方式。流式层先按固定词表给不同标签编码，再对编码对计数，实际使用同一段计数逻辑计算入窗批和驱逐行。基于累计计数算比率，以 OvR 方式复用 AUC。 |
| Pipeline | `numcompute/pipeline.py` 的 `Transformer`、`Estimator`、`Pipeline.fit/predict`、`FeatureUnion.fit/transform` 定义命名、保行和复制语义。旧步骤原地拟合，没有 `partial_fit` 或快照。`tests/test_pipeline.py` 和 `tests/test_optim_pipeline_utils.py` 用替身测试组合。 | **新增**流式 Pipeline。沿用命名步骤、保行与输入保护约定，避免继承旧类的就地拟合与状态。可复用旧 `numcompute/utils.py:iter_batches` 生成内存中的 demo/benchmark 切片。 |
| 输入验证 | `numcompute/_validation.py` 的 `real_array` 检查实数、维数、NaN/Inf，`matrix` 检查二维并转 float64，`integer` 拒绝布尔整数；`fitted` 的检查绑定旧 transformer 的属性。旧 Pipeline 自有 `_matrix` 并把宽度决定权交给步骤。 | **直接复用**基础数值与整数检查，通过新包内部适配层集中旧私有导入。**新增**带参数名的错误、类别 dtype、行数匹配、宽度、转换后溢出与各阶段状态验证。 |
| Benchmark | `numcompute/benchmarking.py` 的 `BenchmarkCase`、`run_benchmarks` 支持检查双方输出、预热、交替顺序和环境记录。`benchmark/run.py` 有 `_loop_welford`、`_loop_histogram` 等旧负载，默认写旧 `benchmark/results.json`。现有 harness 测试含模拟时钟和正确性失败路径。 | **直接复用** harness。**新增**候选分裂的独立循环对照、buffer sweep 驱动及单独的真实测量输出。旧结果文件只作为 2.1 记录。 |
| 树、流管理与绘图 | 旧 `numcompute/` 无分类树、`StreamTrainer` 或 matplotlib 绘图模块。旧 `demo/quickstart.py` 用示例标签演示指标；CSV 样例的列为 `price,weight,category_code`。 | **新增** `tree.py`、`stream.py`、`visualise.py` 和 `demo/stream_demo.ipynb`。准备标签角色明确、含缺失且有足够批次的 CSV。 |

## 2. 包结构与接口

同一发行包保留 `numcompute/` 并包含 `numcompute_stream/`。setuptools discovery 的 `include = ["numcompute*"]` 匹配两个包。运行依赖声明为 `numpy>=1.24` 和 `matplotlib>=3.6`，核心使用标准库和 NumPy，显式导入绘图模块时才加载 pyplot。已在临时 venv 中离线安装 wheel，从仓库外运行导入、训练和绘图。依赖继承自 `aiml`，实际测试版本见阶段 D 记录。

以下接口为**建议设计**。由 2.2 specs 指定的模块、方法和交付物见第 6 节。

| 文件 | 建议公开 API、形状和状态 |
| --- | --- |
| `__init__.py`、`_validation.py` | 前者导出主要组件。后者是私有的数组、整数、标签空间、概率与拟合状态检查。统一报告输入名、期望形状和实际形状。 |
| `io.py` | 重新导出 `load_csv` 和 `iter_csv_chunks`。`iter_supervised_csv(path, *, target_col, chunk_size=1000, delimiter=",", skip_header=0)` 逐批返回 `(X, y)`，有序移出目标列，允许 Python 风格负索引。每个数据批至少两列，特征 NaN 保留，非有限标签或 Inf 特征报错。沿用旧数值读取器的默认精度，`X` 和 `y` 均为 float64。 |
| `preprocessing.py` | **已实现。** `StandardScaler()`、`SimpleImputer(fill_value=0.0)`、`OneHotEncoder(handle_unknown="ignore")` 和 `TabularPreprocessor(numeric_columns, categorical_columns, *, fill_value=0.0)` 提供 `fit(X)`、`partial_fit(X_chunk)`、`transform(X)`。前两者输出 `(n,d)`，Encoder 输出 `(n,p)`，组合器输出 `(n,d_num+p)`。数值累计状态为每列 count、mean 和 M2，scale 和填补值从状态计算。Encoder 只追加类别列。Scaler 和组合器提供 `snapshot()`；组合器的列索引为有序、互斥的非负整数，至少选一列。 |
| `tree.py` | **已实现。** `DecisionTreeClassifier(max_depth=8, min_samples_split=2, max_features=None, buffer_size=128, random_state=0)`，使用 Gini。`fit(X,y)`、`partial_fit(X_chunk,y_chunk)` 返回 self。`predict(X)` 输出 `(n,)`，`predict_proba(X)` 输出 `(n,k)`，列 `j` 对应最近训练的 `classes_[j]`，每行总和为 1。叶概率为训练窗口内该叶的类频率。只读状态含 `is_fitted_`、`pending_samples_`、`retrain_count_` 和 `last_train_samples_`。 |
| `pipeline.py` | **已实现。** `Pipeline(steps)` 接收唯一命名的步骤和末端 `DecisionTreeClassifier`。支持裸树，或一个 `StandardScaler` / `TabularPreprocessor`，前后可串联确定性、逐行独立且保行的无状态 `transform`。增量步骤和树是配置模板，Pipeline 不改写调用方实例。`fit`、`partial_fit`、`predict`、`predict_proba` 形状同树；`transform(X)` 返回发布变换的 `(n,p)`。提供 `reset()`、`is_fitted_`、`classes_`、`pending_samples_`、`retrain_count_`、`last_train_samples_` 和 `last_timing_`。 |
| `stats.py` | **已实现。** `StreamingStats(*, bin_edges)` 使用一组各列共用且固定的严格递增边缘。`update_stats(X_chunk)` 和 `update(X_chunk)` 是同一累计入口。`result(q=0.5, *, ddof=0)` 返回 count、mean、var 各 `(d,)`，quantile 为 `(d,)` 或 `(len(q),d)`，histogram 为 `(d,b)`，edges 为 `(b+1,)`，另有各列 underflow/overflow。`reset()` 清状态，保留边缘配置。有首个非空批之后才可查询结果。 |
| `metrics.py` | **已实现。** `ClassificationMetrics(labels, *, pos_label=None, window_size=None, zero_division=0.0)`。`update(y_true_chunk,y_pred_chunk, *, y_proba=None)` 接收等长 `(n,)` 和可选的 `(n,k)`，概率列按 `labels_` 对齐，行和的绝对容差为 `1e-6`。`result(*, rolling=False)` 返回 `n_samples`、`n_scored`、`accuracy`、逐类 `precision/recall/f1`、`(k,k)` 的 `confusion_matrix`、`auc` 和 `auc_per_class`。给定 `pos_label` 时只计算该类的精确 OvR AUC，否则对已定义类作宏平均。滚动查询要求配置窗口。`reset()` 清空数据。 |
| `stream.py` | **已实现。** `StreamTrainer(pipeline, *, labels, pos_label=None, window_size=None)`。`score_chunk(X,y)` 返回批状态、已评估数和版本、`metrics` 及 `y_pred (n,)`、对齐后的 `y_proba (n,k)`；warmup 无评估行，accuracy 与 AUC 为 NaN。`fit_chunk(X,y)` 先评估再学习，成功时附加 `log` 摘要并写入 `logs`。空批的 `log` 为 None。`reset()` 清空 Pipeline、累计指标与日志。 |
| `visualise.py` | **已实现。** `plot_metric_over_time(metric_values,title,ylabel)` 接收一维时间序列，NaN 保留空点。`compare_models(metric1,metric2,labels)` 接收等长序列与两个模型名。`plot_predictions_vs_ground_truth(y_true,y_pred)` 接收等长一维标签，真值和预测共用离散刻度。三个函数接受关键字 `save_path=None, show=False` 并返回 `(Figure, Axes)`，由调用方负责关闭 Figure。 |

**形状与状态约定。** 输入一律为二维 `X=(n_samples,n_features)`、一维 `y=(n_samples,)`，不自动重排单样本或压平 `(n,1)`。统计和预处理沿 `axis=0` 按列累计，变换保留行数。数值路径接受实数 array-like 并转换成 float64，拒绝不支持的 dtype；独立 Encoder 可接收同一 dtype kind 的文本或有限实数，表格组合器用数值类别码。标签采用同一 dtype kind 的有限实数、布尔或字符串，可在同一 kind 内扩大位宽，避免字符串截断和跨 signed/unsigned 强制合成 float64。字段类型错误用 `TypeError`，形状、宽度、非法值和参数范围用 `ValueError`，未发布模型的直接推理用 `RuntimeError`。更新首先验证配置、数组、宽度和数值；接受宽度匹配的零行更新作为 no-op，首次零行更新不固定宽度。批量 `fit` 要求非空，成功时以输入重新拟合并重置旧统计、类别、缓冲、重训计数和 RNG；已拟合后的零行预测返回形状正确的空数组。方法不改写调用方数组，需保留的批次使用独立副本。

## 3. 模型、缓冲和预处理版本

### 树和流式时序

1. `fit` 无视缓冲门槛，从全部当前输入训练一棵新树。独立树的 `partial_fit` 保存数值特征及标签，达到门槛后调用与 `fit` 相同的建树路径。后续 Pipeline 可持有一份原始数据缓冲，在触发时对新候选树调用公开的 `fit`，并在 Pipeline 层记录重训版本和次数。独立树与 Pipeline 缓存的数据不同，本阶段将缓冲逻辑留在树内，避免为尚未实现的调用方设计通用调度器。
2. 每次非空更新整块追加。总数 `>= buffer_size` 时，用所有已缓冲样本完整重训一次并清空，包含超出门槛的整批尾部。记录实际训练样本数。两次重训间继续用旧树。流结束保留不足门槛的待训练样本，以便同一 stream 继续；demo 和 benchmark 报告余量，初版不加自动 flush。
3. 树未就绪时直接 `predict/predict_proba` 报错，`score_chunk` 给出 `warmup`、零评估行和 NaN 指标。`fit_chunk` 先按发布版本评估全批并暂存结果，再用本批学习，成功后提交指标与日志。即使当前批首次触发重训，整批仍属 warmup。启动批只增加训练样本计数。从下一批开始，指标分母只含真正预测过的样本。`score_chunk` 可重复调用，日志、模型和累计指标均不变。
4. 候选树、累计预处理状态和快照先构建在独立状态，全部成功后一次替换。输入错误或候选构建失败时，保留旧快照、旧缓冲及计数，本批可重试一次且不会重复累计。仅在单线程使用下承诺该提交语义。

### 坐标一致性

- Pipeline 用两套状态。学习状态在每个有效 chunk 更新，保存到目前为止的特征统计及类别字典。发布状态绑定一棵树，持有该树训练所用的只读预处理快照，快照包含填补值、均值、scale、类别键、输出列顺序和宽度。预测只读发布状态。
- 组合器数值分支用**同一份原始观测的 RunningStats** 同时得到均值填补值、中心和 scale。实际变换先填补后标准化。全 NaN 列使用固定 fallback、中心 0 和 scale 1，首个观测到达后转正常统计；已观测的常量列用 scale 1。快照保存只读数值参数及该版本的类别映射，后续学习更新通过替换状态发布。
- 缓冲始终仅存原始数据，包含原始 NaN 和类别码。达到门槛时先更新本批学习状态，再冻结候选快照；拼接并用这一份快照**重新变换所有缓冲行**，构建候选树。随后成对发布快照和树。例如旧中心为 0，新数据使中心变为 10，旧树期间 (X=5) 仍按中心 0 变换；新树和所有重训行统一按中心 10 变换。
- Encoder 以 `(输入列号, 类别值)` 为键。每批向量化发现新键，按列索引、类别值排序后作为一个后缀追加；同一批之前的列索引永不移动。数值块固定在前，编码列固定在后。出现新类别但尚未重训时，旧快照仍产生旧宽度，未知类别在对应旧列编码为全零。重训时同一新版映射编码所有原始缓冲行，旧列意义仍在原位。独立 Encoder 可配置 `error` 且报告未知类别；Pipeline 的表格组合器使用 `ignore`，允许真实流出现新类别。
- Pipeline 串联无状态变换和终端模型；多个依赖不同变化坐标的学习步骤须封装成定义清楚学习输入的组合器。初版的 `TabularPreprocessor` 使用独立数值/类别输入分支，避免把普通 `partial_fit + transform` 串联误当作一致性保证。代价是保留原始缓冲、复制预处理快照以及在重训时重新编码；dense one-hot 在高基数时耗内存。

### 阶段 C 集成

- Pipeline 只维护一份原始 `(X, y)` 缓冲。达到门槛时，对整个缓冲应用同一候选快照，再调用候选树的 `fit`。
- 每次候选树 `fit` 会把树内 `retrain_count_` 设回 1。Pipeline 自己维护累计重训次数，在候选成功发布时递增。
- 候选预处理快照与候选树成对发布。树构建失败时，保留旧发布对、已有原始缓冲和累计计数。独立 `StandardScaler` 和 `TabularPreprocessor` 已提供冻结的 `transform` 接口。
- 本实现的增量步骤限于上述两类，依赖其学习状态在成功更新后整体替换的契约。两类新增内部清空方法供 Pipeline 重置。组合器负责均值填补和类别编码，独立 Imputer 和 Encoder 继续通过各自接口使用。Scaler 路径拒绝 NaN 进入待训缓冲。

### 分裂与数值边界

- 根深度为 0；`max_depth=0`、纯节点、样本数小于 `min_samples_split`、无有效正增益或常量特征时出叶。每个候选特征排序，只在相邻不同值之间评估切分；以左组最大观测值作为阈值，按 `<=` 分左边，避免算 midpoint 的溢出和舍入。区间内新值按这个明确的阈值规则路由。
- 每个候选特征用类计数前缀和、广播及向量化 Gini 计算所有合法候选。增益 `<= 1e-12` 视作无有效分裂；候选增益距全局最大值 `<= 1e-12` 时按最小特征索引、再按最小阈值决胜。预测票数相同时选排序后最前的类别。单一类别可训练，整棵树为叶时概率仍有效。
- `max_features=None` 用全部列；正整数 `m` 要求 `1 <= m <= p`；浮点 `f` 要求 `0 < f <= 1`，每节点用 `ceil(f*p)` 列，`p` 为当前树的变换后宽度。在每个拟分裂节点无放回抽一次，遍历候选时按特征索引排序。显式 `random_state` 为非负整数，每次完整建树重新初始化 NumPy Generator，固定左后右的节点访问顺序；相同输入、种子、配置、NumPy 环境可复现。记录种子和版本。采样特征均无法分裂时直接使用叶节点。
- 独立树在特征转换成 float64 后拒绝 NaN、Inf 和转换溢出。极大整数特征可能因转换丢失精度；类别标签保留 NumPy dtype。组合器在数值分支填补 NaN，并拒绝非有限类别码。矩估计复用 Welford/Chan 的中心矩，候选统计若出现非有限均值、非有限方差或负方差则拒绝更新，保留旧状态。标准化溢出抛出 `ValueError`。
- 节点中候选特征逐个处理，先保存每列最大增益，再重算第一个满足平局容差的特征，选择最小合格阈值。单节点 `n` 行、`k` 类、`m` 个候选特征的时间约为 `O(m(n log n + nk))`，计数工作区约 `O(nk)`。建树用显式栈执行递归划分，逐特征处理及按节点批量预测路由保留循环。节点数据和形状依赖上一层划分，逐特征还可避免同时分配所有候选的计数数组。CSV、chunk 调度、类别映射、窗口驱逐与绘图允许必要循环；矩、掩码、计数、比较等数组运算使用 NumPy。

## 4. 统计、指标和内存

- **均值、方差、分位数。** 按列忽略 NaN，`RunningStats` 的持久矩状态为 `O(d)`，`count <= ddof` 或全缺失列返回 NaN。分位数初版精确，按块保留已看数值矩阵，在查询时调用旧 `quantile(axis=0, method="linear", nan_policy="omit")`。历史 `N` 行占 `O(Nd)`，查询最坏 `O(dN log N)`，拼接还有 `O(Nd)` 临时内存。Stats 是独立组件，由 demo 选用，Scaler 不保存分位数历史。
- **直方图。** 对每列用相同固定边缘累加计数，`O(db)` 持久存储；末箱含右端点。范围外分别计 underflow/overflow，NaN 不计。固定边缘使跨批计数可加，且便于与离线参考核对。初版提供累计，sliding histogram 为 specs 中的可选项。
- **累计与滚动分类指标。** Trainer 配置固定标签表，训练树在每个缓冲区可只见其中部分标签。固定表用于混淆矩阵及日志；未在树中出现的已配置类别以零概率列对齐，表外标签在 Trainer 学习前报错。每批混淆矩阵加到 `int64` 总矩阵，accuracy 为总对角和除以总评估行数，从行/列总数提取 precision、recall 和 F1。默认零分母用 0，尚无评估行时 accuracy 为 NaN。滚动窗口按**样本**保存最近 `W` 条真值、预测和可用概率，驱逐可切开 chunk；对子矩阵加新计数、减驱逐计数。训练 warmup 不占窗口。
- **AUC。** 有概率的评估行按评估当时的值保存，树更新后不回算。精确累计 AUC 用旧二分类 kernel 计算一对多，平局计半；rolling AUC 对 window 内有概率的行查询。只有正类或只有负类（包括空集）时该类 AUC 为 NaN，宏平均只含已定义的类别，无定义类别则为 NaN。单一真实类别的窗口所有 OvR AUC 都为 NaN。基于 `M` 行概率、`k` 类的累计存储为 `O(Mk)`；查询单个类别 `O(M log M)`，全类别 `O(kM log M)`，窗口以 `W` 为上界。查全量曲线反复排序的成本应与训练计时分开报告。缺概率的行仍进入计数和窗口，并从 AUC 样本中排除；结果返回概率行数供解释。多模型版本概率混合的 AUC 表示整个预序列预测历史。
- **内存。** Trainer 每批成功后记录当前拥有的 ndarray buffer 字节数，按底层存储去重，分列报告树/预处理快照、学习状态、待训样本和指标历史；指向同一数组的引用只计一次，object 等间接对象及 Python 节点、容器、日志对象、解释器、绘图和临时工作区不在此口径。单独的 StreamingStats 历史由调用方持有，可单列其数组估计。该值是批末的实现内存估计，训练期间峰值和进程 RSS 需另行定义与实测；跨 buffer 比较使用同一口径。
- **计时与日志。** `preprocessing_s` 累计本批学习和实际变换，`tree_fit_s` 包含候选树的输入验证与建树，`retrain_s` 从合并缓冲起覆盖冻结、整缓冲变换、拟合和发布，`update_s` 覆盖 Pipeline 的校验与整个学习调用。Trainer 另记 `evaluation_s`。累计或滚动 AUC 查询与日志构造在这些计时之外。日志存 accuracy、宏 precision/recall/F1、AUC 及行数等标量；完整逐类结果和矩阵由评分或指标查询返回。

## 5. Benchmark、demo 与实施顺序

**循环与向量化。** `benchmark.run_stream` 复用旧 harness 比较生产 `_candidate_gains` 与独立 Python 循环。输入相同，计时包括单特征排序和所有候选计分。计时前核对全部阈值、增益及最优阈值分区。数据生成、选择检查、I/O 和绘图均在计时外。每个实现预热 2 次，测 7 组各 5 次，交替执行顺序。JSON 记录原始样本、最小值、中位数、最大值与环境。本实验的耗时范围为单特征候选 kernel。

**Buffer sweep。** 64、128、256 使用相同的有序 CSV、32 行 chunks、预处理、树参数、滚动窗口和种子。三个模型在所有共同可评估批次上比较按样本累计的 accuracy，并记录各自全可评估段的 accuracy、覆盖率、首次评分批次、训练行数和待训尾部。每个模型预热 1 次，测 3 次，每次轮换运行顺序并核对结果一致。分开记录 `tree_fit_s`、`retrain_s` 和 `overall_s`。整体时间含新建 Trainer、评分、在线更新、AUC 查询及内存日志。CSV 读取、绘图和实验结果序列化在计时外。AUC 在 notebook 展示，sweep 的共同批次比较使用 accuracy。速度只作为测量结果记录。

**Demo。** `demo/stream_workflow.py` 为 Notebook 和 benchmark 配置相同的 Pipeline 与 StreamTrainer，并从 `numcompute_stream.io.iter_supervised_csv` 读取原序 chunks。`fit_chunk` 先评分，再调用 `pipeline.partial_fit`。Notebook 用独立 `StreamingStats` 统计两个原始数值列，展示三类图和学习前预测。64 与 256 的比较图使用两个真实运行及同一个共同批次掩码。数据和生成规则见 [CSV 数据说明](../demo/data/README.md)。

| 顺序 | 实现与验证 |
| --- | --- |
| 1 | **已完成。** 建立包入口，实现本阶段需要的列索引、数值及维度验证适配和 CSV 包装，并验证 T01–T03。模型状态和重置契约在对应阶段实现。 |
| 2 | **阶段 A 已完成。** 实现独立 Gini 树和阈值缓冲重训，验证叶概率、分裂、随机性、精确达标、整批越线与失败保护。 |
| 3 | **阶段 B 已完成。** 复用 `RunningStats` 实现增量预处理和独立 StreamingStats，验证快照、NaN、统计与列顺序。 |
| 4 | **阶段 C 已完成。** 在树及预处理之上接 Pipeline 的原始缓冲与成对发布，验证两次重训之间的坐标。 |
| 5 | **阶段 C 已完成。** 实现指标累计、滚动和 AUC，接入 Trainer 并验证先评估后学习、失败重试与日志。 |
| 6 | **阶段 D 已完成。** 执行绘图、Notebook 和两类 benchmark，验证旧套件、2.2 测试与离线 wheel。 |
| 7 | **阶段 E 本地材料已就绪。** 生成可编辑 Word、3 页 PDF 和视频录制材料。正式导出、录制、push 和提交见 [最终提交清单](submission_checklist.md)。 |

### 第 1 阶段完成记录

新增 `numcompute_stream/__init__.py`、`_validation.py`、`io.py` 和 `tests/test_stream_io.py`。包入口导出两个旧读取器及新的 `iter_supervised_csv`。适配层复用旧 `integer` 和 `real_array`，补目标列负索引和带输入名的验证错误。

在 `aiml` 环境实测旧读取器会跳过空行、保留尾批，并分别由 NumPy 和旧迭代器检查批内及批间列数。空标签在 `missing="propagate"` 下产生 NaN，文本字段会触发旧解析器的 `ValueError`。新包装层在产出前拒绝 NaN/Inf 标签、Inf 特征及缺少特征列的文件。实现遵循原计划，并明确了两组数组沿用旧读取器的 float64 输出。用现有 setuptools 的 discovery 验证当前规则可找到两个包，wheel 验证留在第 6 阶段。

在仓库根目录用现有 `aiml`（Python 3.12.13、NumPy 2.5.2、pytest 9.2.0）执行如下命令。

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/aiml/bin/python -B -m unittest discover -s tests -p test_stream_io.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/aiml/bin/python -B -m pytest -p no:cacheprovider tests --ignore=tests/test_stream_io.py
```

T01–T03 共 **3 项通过**（0.012 s）。旧套件 **642 项通过，0 项失败，4 条原有 NumPy 配置输出警告**（0.59 s）。两条命令退出码均为 0。

### 阶段 A 完成记录

新增 `numcompute_stream/tree.py` 与 `tests/test_stream_tree.py`，扩展输入验证与包导出。候选阈值使用类别计数前缀和批量计算 Gini 增益，建树和预测均按节点遍历。训练成功后才发布树与类别并清空缓冲。未训练时直接预测报 `RuntimeError`，合法空增量不改变状态，已训练时零行推理返回正确形状。

在同一 `aiml` 环境分别运行树测试 **22 项通过**、流式 I/O 测试 **3 项通过**、旧回归 **642 项通过**。全量合跑 **667 项通过，0 项失败**，仍有 4 条旧 NumPy 配置输出警告。树测试包括 T04–T22，以及重训失败重试、大整数标签和独立多类 Gini 参考计算，完整命令见 README。

### 阶段 B 完成记录

新增 `preprocessing.py`、`stats.py` 和两个对应测试文件。各组件先验证候选状态再赋值。快照只保存变换参数，类别映射为只读且按版本替换。组合器接受数值表，类别码统一为 float64，大整数码可能合并；直接使用独立 Encoder 时保留原有类别 dtype。直方图为累积固定箱，精确分位数保留 `O(Nd)` 历史。极端有限输入如果使浮点矩溢出，会显式报错。

在同一 `aiml` 环境运行预处理 **12 项**和统计 **4 项**，全部通过。全量套件 **683 项通过，0 项失败**，含阶段 B 16 项、之前的 2.2 测试 25 项、2.1 回归 642 项。警告是既有的 4 条 NumPy 配置输出。

### 阶段 C 完成记录

新增 `pipeline.py`、`metrics.py`、`stream.py` 和三个对应测试文件。对已训练批只调用一次概率预测，并按树类别顺序选出预测，同步映射到配置词表。独立树实现保持原样。全量套件 **704 项通过、0 项失败**，其中阶段 C 21 项（Pipeline 8、指标 7、Trainer 6）、既有 2.2 测试 41 项、2.1 回归 642 项。4 条原有 NumPy 配置输出警告。

### 阶段 D 完成记录

现有 `aiml` 环境使用 Python 3.12.13、NumPy 2.5.2、matplotlib 3.11.1。在获得本地 IPC 权限后，执行器用实际 `aiml` Jupyter kernel 运行了 [Notebook](../demo/stream_demo.ipynb) 的 5 个代码单元，保存 3 个嵌入图及相应 PNG。64 行 buffer 的 Notebook 输出包含 31 次重训、16 行待训、1,936 条学习前预测，累计 accuracy 约 0.780、AUC 约 0.816。合成 CSV 用固定种子再生成后逐字节一致。

轮子检查用 `pip wheel --no-index --no-deps --no-build-isolation` 从临时源码副本构建 `numcompute-0.1.0-py3-none-any.whl`，离线装入临时 venv。仓库外通过 wheel 导入两包，核对依赖声明，并通过流式训练、预测和 Agg PNG 检查。全量测试 **709 项通过、0 项失败**，其中 2.1 回归 642 项，2.2 测试 67 项（阶段 D 新增 5 项），另有 4 条既有 NumPy 配置输出警告。README 保留可重跑命令。

#### 可引用的实测结果

以下为 2026 年 11 月 18 日在 macOS arm64 的 `aiml` 环境、种子 2026 下的中位数。分裂实验的阈值、增益及最优分区检查均通过。未取整数值、原始重复样本、数据 SHA-256 和运行时代码哈希见 [stream_results.json](../benchmark/stream_results.json)。

| 单特征候选行数 | 候选阈值数 | NumPy／µs | Python 循环／µs | 循环除以 NumPy |
| --- | ---: | ---: | ---: | ---: |
| 256 | 181 | 32.01 | 242.57 | 7.58× |
| 2,048 | 454 | 127.59 | 1,003.99 | 7.87× |

| Buffer | 首次评分批 | 覆盖率 | 共同准确率 | 重训次数 | 待训行 | Tree fit／s | Retrain／s | Overall／s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3 | 96.80% | 77.81% | 31 | 16 | 0.0302 | 0.0313 | 0.0639 |
| 128 | 5 | 93.60% | 79.36% | 15 | 80 | 0.0226 | 0.0232 | 0.0561 |
| 256 | 9 | 87.20% | 76.43% | 7 | 208 | 0.0147 | 0.0150 | 0.0475 |

共同评估为第 9–63 批的 1,744 行。128 的准确率在这次共同评估上最高，256 的整体时间最短且评估覆盖率最低。测量只反映此合成数据、硬件与计时范围。阶段 E 的报告和视频材料引用了这两张表与 Notebook 输出。

### 阶段 E 最终核对

`aiml` 全量复跑为 709 项通过、0 项失败，含 642 项旧回归和 67 项 2.2 测试。SHA-256 核对覆盖 benchmark 记录的 9 个源文件和合成 CSV。相关程序和 Notebook 与 D 阶段一致，原有的真实 kernel 与离线 wheel 验证继续适用。可编辑报告和本地 PDF 已生成，录制和正式提交仍待人工完成。[提交清单](submission_checklist.md)

## 6. 需求覆盖与待确认

| 老师明确要求 | 实施证据及对应评分 |
| --- | --- |
| 从零实现限深单棵分类决策树、按整缓冲重训、两次间复用最近树；NumPy 向量化和边界行为 | 第 2–3 节及树、缓冲、数值测试，报告解释 Gini、复杂度和必要循环。Algorithm Design 25。 |
| `stream.py` 的 `fit_chunk/score_chunk` 和日志；增量预处理、`update_stats`、指标 `update/reset/result`、累计混淆矩阵和 AUC、rolling metrics、流式 Pipeline | 第 2–4 节及 API、集成测试。完整 `numcompute_stream/`，只用 plain Python、NumPy、matplotlib。Functionality Implementation 25。 |
| 可复用及可维护代码 | 保留旧包、复用表中的计算 kernel 与 harness；集中校验、状态契约与失败语义；验证同一 wheel 可安装导入。Code Reusability 15，Code Maintainability 15。 |
| 至少 30 项有意义的单元测试，兼顾正常及流式边界；循环/向量化和 buffer_size 双 benchmark，实验真实测量 | [测试计划](test_plan.md)、现有基线及后续运行记录、正确性检查和实测结果。Testing & Debugging Evidence 10。 |
| CSV 经自定义 `io.py` 按批读取；`demo/stream_demo.ipynb` 经 `partial_fit` 学习并通过 `visualise.py` 的三个指定函数作图，可保存或显示 | 第 2、5 节，保留 notebook 工作流和可自动核对的输出。 |
| README、最多 5 分钟视频、最多 4 页 PDF（决策、测试及边界、benchmark、反思）、fork 与个人分支 | README 说明安装、形状、状态与限制；报告引用实际测试和 benchmark，视频演示完整流。Coding Style、Documentation、Comments 10。后续核对 fork 关联和分支。 |

以上未由规格定死的接口和策略属于**建议设计**，尤其是尾部不强制重训、精确 AUC/分位数、固定标签表、数值类别 CSV、估算内存以及组合器快照。可据当前文字直接推进，暂未发现阻塞性的规格歧义。交付前需要核验个人 `origin` 在 GitHub 上是否确实关联小组上游，以及课程提交入口对 PDF、视频链接的具体提交方式；当前仅看到了仓库本地信息和用户提供的规格。
