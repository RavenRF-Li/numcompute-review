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
| 统计 | `numcompute/stats.py` 的 `RunningStats.update/merge/copy` 已支持分批矩估计和 NaN 策略。现有 `histogram` 展平后单次计数，`quantile` 对传入数组单次归约。 `tests/test_stats_metrics.py` 对比不等长分批与批量矩，并覆盖 merge；`tests/test_stats_metrics_edges.py` 有方差精度边界。 | **直接复用** `RunningStats` 的累计计算、`histogram` 的固定边缘计数与 `quantile` 的精确归约。**新增** `update_stats`、按列直方图计数和原始分位数样本管理。 |
| 指标 | `numcompute/metrics.py` 中 `confusion_matrix` 支持显式标签顺序，内部用 `bincount`；accuracy、precision、recall、F1 均从输入批计算。`roc_auc_score` 是精确二分类，合并同分，单一真实类别会报错。现有测试覆盖顺序、平局和未定义的分母。 | **直接复用**显式标签顺序的混淆矩阵和二分类 AUC kernel。**新增**累加和逐样本 rolling window，基于累计计数算比率，以 OvR 方式复用 AUC。 |
| Pipeline | `numcompute/pipeline.py` 的 `Transformer`、`Estimator`、`Pipeline.fit/predict`、`FeatureUnion.fit/transform` 定义命名、保行和复制语义。旧步骤原地拟合，没有 `partial_fit` 或快照。`tests/test_pipeline.py` 和 `tests/test_optim_pipeline_utils.py` 用替身测试组合。 | **新增**流式 Pipeline。沿用命名步骤、保行与输入保护约定，避免继承旧类的就地拟合与状态。可复用旧 `numcompute/utils.py:iter_batches` 生成内存中的 demo/benchmark 切片。 |
| 输入验证 | `numcompute/_validation.py` 的 `real_array` 检查实数、维数、NaN/Inf，`matrix` 检查二维并转 float64，`integer` 拒绝布尔整数；`fitted` 的检查绑定旧 transformer 的属性。旧 Pipeline 自有 `_matrix` 并把宽度决定权交给步骤。 | **直接复用**基础数值与整数检查，通过新包内部适配层集中旧私有导入。**新增**带参数名的错误、类别 dtype、行数匹配、宽度、转换后溢出与各阶段状态验证。 |
| Benchmark | `numcompute/benchmarking.py` 的 `BenchmarkCase`、`run_benchmarks` 支持检查双方输出、预热、交替顺序和环境记录。`benchmark/run.py` 有 `_loop_welford`、`_loop_histogram` 等旧负载，默认写旧 `benchmark/results.json`。现有 harness 测试含模拟时钟和正确性失败路径。 | **直接复用** harness。**新增**候选分裂的独立循环对照、buffer sweep 驱动及单独的真实测量输出。旧结果文件只作为 2.1 记录。 |
| 树、流管理与绘图 | 旧 `numcompute/` 无分类树、`StreamTrainer` 或 matplotlib 绘图模块。旧 `demo/quickstart.py` 用示例标签演示指标；CSV 样例的列为 `price,weight,category_code`。 | **新增** `tree.py`、`stream.py`、`visualise.py` 和 `demo/stream_demo.ipynb`。准备标签角色明确、含缺失且有足够批次的 CSV。 |

## 2. 包结构与接口

保留 `numcompute/`，在同一发行包内加入 `numcompute_stream/`。现有 setuptools discovery 的 `include = ["numcompute*"]` 匹配两个包，使用带 `__init__.py` 的普通包即可。实现阶段在个人分支补元数据和 matplotlib 依赖，核心仅用标准库、NumPy，绘图模块用 matplotlib；顶层导入不提前加载 pyplot。保留现有 NumPy/Python 版本约束，核验 matplotlib 版本下限与支持矩阵。后续先验证 editable install，再构建并在独立环境安装 wheel，从仓库外导入两个包并运行最小流式示例。跨包复用随同一 wheel 分发。

以下接口为**建议设计**。由 2.2 specs 指定的模块、方法和交付物见第 6 节。

| 文件 | 建议公开 API、形状和状态 |
| --- | --- |
| `__init__.py`、`_validation.py` | 前者导出主要组件。后者是私有的数组、整数、标签空间、概率与拟合状态检查。统一报告输入名、期望形状和实际形状。 |
| `io.py` | 重新导出 `load_csv` 和 `iter_csv_chunks`。`iter_supervised_csv(path, *, target_col, chunk_size=1000, delimiter=",", skip_header=0)` 逐批返回 `(X, y)`，有序移出目标列，允许 Python 风格负索引。每个数据批至少两列，特征 NaN 保留，非有限标签或 Inf 特征报错。沿用旧数值读取器的默认精度，`X` 和 `y` 均为 float64。 |
| `preprocessing.py` | `StandardScaler()`、`SimpleImputer(fill_value=0.0)` 和 `OneHotEncoder(handle_unknown="ignore")` 提供 `fit(X)`、`partial_fit(X_chunk)`、`transform(X)`；前两者输出 `(n,d)`，后者输出 `(n,p)`。Scaler 保存每列 count、mean、M2 和 scale；Imputer 保存每列观测计数和当前填补值；Encoder 保存只追加的类别映射及输出宽度。 `TabularPreprocessor(numeric_columns, categorical_columns, *, fill_value=0.0)` 按列分支，输出 `(n,d_num+p)`，支持相同接口及 `snapshot()`，列列表有序、互斥，合计至少一列。 |
| `tree.py` | `DecisionTreeClassifier(max_depth=8, min_samples_split=2, max_features=None, buffer_size=128, random_state=0)`，使用 Gini。`fit(X,y)`、`partial_fit(X_chunk,y_chunk)` 返回 self。`predict(X)` 输出 `(n,)`，`predict_proba(X)` 输出 `(n,k)`，列 `j` 对应最近训练的 `classes_[j]`，每行总和为 1。叶概率为训练窗口内该叶的类频率。只读状态含 `is_fitted_`、`pending_samples_`、`retrain_count_` 和最近训练行数。 |
| `pipeline.py` | `Pipeline(steps)` 接收唯一命名的步骤和末端 `DecisionTreeClassifier`。中间允许一个自带 `snapshot()` 的增量预处理步骤（可组合数值和类别列），前后可接确定性、保行的无状态 `transform`，或仅用树。`fit`、`partial_fit`、`predict`、`predict_proba` 对外形状同树；`transform(X)` 返回当前已发布快照的 `(n,p)`。维护原始输入宽度、学习中的预处理状态、发布状态、原始样本缓冲和重训计数。 |
| `stats.py` | `StreamingStats(*, bin_edges)` 使用一组各列共用且固定的严格递增边缘。`update_stats(X_chunk)` 和 `update(X_chunk)` 是同一累计入口。`result(q=0.5, *, ddof=0)` 返回 count、mean、var 各 `(d,)`，quantile 为 `(d,)` 或 `(len(q),d)`，histogram 为 `(d,b)`，edges 为 `(b+1,)`，另有各列 underflow/overflow。`reset()` 清状态，保留边缘配置。 |
| `metrics.py` | `ClassificationMetrics(labels, *, pos_label=None, window_size=None, zero_division=0.0)`。`update(y_true_chunk,y_pred_chunk, *, y_proba=None)` 接收等长 `(n,)` 和可选的 `(n,k)`，概率列按配置的 `labels` 对齐。`result(*, rolling=False)` 返回样本数、准确率、逐类 precision/recall/F1、`(k,k)` 混淆矩阵、AUC 与有概率行数；给定 `pos_label` 时返回该类的精确 OvR AUC，省略时返回已定义 OvR AUC 的宏平均和逐类结果。`reset()` 清空累计和滚动状态。 |
| `stream.py` | `StreamTrainer(pipeline, *, labels, pos_label=None, window_size=None)`。`score_chunk(X,y)` 纯评估并返回批指标及本批预测数组（标签 `(n,)`、对齐概率 `(n,k)`），启动时返回 warmup、NaN 及零行预测。`fit_chunk(X,y)` 执行先评估后学习，返回评估结果与含本批/累计/滚动指标、样本数、评估和发布版本、重训时间及内存的摘要。日志仅保留摘要，调用方可用本批返回的预测作图。空批返回空结果，不记日志；`reset()` 恢复未训练状态并清空指标和日志，保留配置。 |
| `visualise.py` | `plot_metric_over_time(metric_values,title,ylabel)` 接收一维时间序列，NaN 表示缺点；`compare_models(metric1,metric2,labels)` 接收等长序列与两个模型名；`plot_predictions_vs_ground_truth(y_true,y_pred)` 接收等长一维标签，共用离散类刻度。都附加仅限关键字的 `save_path=None, show=False`，返回 `(Figure, Axes)`；调用方可选择保存、显示，并负责关闭 Figure。 |

**形状与状态约定。** 输入一律为二维 `X=(n_samples,n_features)`、一维 `y=(n_samples,)`，不自动重排单样本或压平 `(n,1)`。统计和预处理沿 `axis=0` 按列累计，变换保留行数。数值路径接受实数 array-like 并转换成 float64，拒绝不支持的 dtype；独立 Encoder 可接收同一 dtype kind 的文本或有限实数，表格组合器用数值类别码。标签采用同一 dtype kind 的有限实数、布尔或字符串，可在同一 kind 内扩大位宽，避免字符串截断和跨 signed/unsigned 强制合成 float64。字段类型错误用 `TypeError`，形状、宽度、非法值和参数范围用 `ValueError`，未发布模型的直接推理用 `RuntimeError`。更新首先验证配置、数组、宽度和数值；接受宽度匹配的零行更新作为 no-op，首次零行更新不固定宽度。批量 `fit` 要求非空，成功时以输入重新拟合并重置旧统计、类别、缓冲、重训计数和 RNG；已拟合后的零行预测返回形状正确的空数组。方法不改写调用方数组，需保留的批次使用独立副本。

## 3. 模型、缓冲和预处理版本

### 树和流式时序

1. `fit` 无视缓冲门槛，从全部当前输入训练一棵新树。独立树的 `partial_fit` 保存数值特征及标签，Pipeline 则保存原始特征及标签。两条入口共享一个内部缓冲调度和建树路径。Pipeline 只在缓冲触发时交付变换后的完整训练集用于构建候选树，末端树自身的公共 `partial_fit` 仍可独立工作。Pipeline 内维持一份缓冲。
2. 每次非空更新整块追加。总数 `>= buffer_size` 时，用所有已缓冲样本完整重训一次并清空，包含超出门槛的整批尾部。记录实际训练样本数。两次重训间继续用旧树。流结束保留不足门槛的待训练样本，以便同一 stream 继续；demo 和 benchmark 报告余量，初版不加自动 flush。
3. 树未就绪时直接 `predict/predict_proba` 报错，`score_chunk` 给出 `warmup`、零评估行和 NaN 指标。`fit_chunk` 先按发布版本评估全批并暂存结果，再用本批学习，成功后提交指标与日志。即使当前批首次触发重训，整批仍属 warmup。启动批只增加训练样本计数。从下一批开始，指标分母只含真正预测过的样本。`score_chunk` 可重复调用，日志、模型和累计指标均不变。
4. 候选树、累计预处理状态和快照先构建在独立状态，全部成功后一次替换。输入错误或候选构建失败时，保留旧快照、旧缓冲及计数，本批可重试一次且不会重复累计。仅在单线程使用下承诺该提交语义。

### 坐标一致性

- Pipeline 用两套状态。学习状态在每个有效 chunk 更新，保存到目前为止的特征统计及类别字典。发布状态绑定一棵树，持有该树训练所用的只读预处理快照，快照包含填补值、均值、scale、类别键、输出列顺序和宽度。预测只读发布状态。
- 数值分支在**同一份原始已观测值**上更新 Imputer 和 Scaler，从而避免把随时间漂移的填充值累计进 Scaler。实际变换先填补后标准化。全 NaN 列的 Imputer 使用固定 fallback，Scaler 在零观测时用中心 0、scale 1；首个观测到达后转正常统计。已观测的常量列用 scale 1。该统计约定独立于传入顺序，是针对增量学习的设计选择。
- 缓冲始终仅存原始数据，包含原始 NaN 和类别码。达到门槛时先更新本批学习状态，再冻结候选快照；拼接并用这一份快照**重新变换所有缓冲行**，构建候选树。随后成对发布快照和树。例如旧中心为 0，新数据使中心变为 10，旧树期间 (X=5) 仍按中心 0 变换；新树和所有重训行统一按中心 10 变换。
- Encoder 以 `(输入列号, 类别值)` 为键。每批向量化发现新键，按列索引、类别值排序后作为一个后缀追加；同一批之前的列索引永不移动。数值块固定在前，编码列固定在后。出现新类别但尚未重训时，旧快照仍产生旧宽度，未知类别在对应旧列编码为全零。重训时同一新版映射编码所有原始缓冲行，旧列意义仍在原位。独立 Encoder 可配置 `error` 且报告未知类别；Pipeline 的表格组合器使用 `ignore`，允许真实流出现新类别。
- Pipeline 串联无状态变换和终端模型；多个依赖不同变化坐标的学习步骤须封装成定义清楚学习输入的组合器。初版的 `TabularPreprocessor` 使用独立数值/类别输入分支，避免把普通 `partial_fit + transform` 串联误当作一致性保证。代价是保留原始缓冲、复制预处理快照以及在重训时重新编码；dense one-hot 在高基数时耗内存。

### 分裂与数值边界

- 根深度为 0；`max_depth=0`、纯节点、样本数小于 `min_samples_split`、无有效正增益或常量特征时出叶。每个候选特征排序，只在相邻不同值之间评估切分；以左组最大观测值作为阈值，按 `<=` 分左边，避免算 midpoint 的溢出和舍入。区间内新值按这个明确的阈值规则路由。
- 每个候选特征用类计数前缀和、广播及向量化 Gini 计算所有合法候选。增益 `<= 1e-12` 视作无有效分裂；候选增益距全局最大值 `<= 1e-12` 时按最小特征索引、再按最小阈值决胜。预测票数相同时选排序后最前的类别。单一类别可训练，整棵树为叶时概率仍有效。
- `max_features=None` 用全部列；正整数 `m` 要求 `1 <= m <= p`；浮点 `f` 要求 `0 < f <= 1`，每节点用 `ceil(f*p)` 列，`p` 为当前树的变换后宽度。在每个拟分裂节点无放回抽一次，遍历候选时按特征索引排序。显式 `random_state` 为非负整数，每次完整建树重新初始化 NumPy Generator，固定左后右的节点访问顺序；相同输入、种子、配置、NumPy 环境可复现。记录种子和版本。采样特征均无法分裂时直接使用叶节点。
- 独立树拒绝非有限值；Pipeline 在数值分支填补 NaN。特征 Inf、标签 NaN/Inf 和非法类别值在更新前拒绝。所有数值路径复核 float64 转换与运算结果；矩估计和标准化检查已观测列的非有限状态，识别溢出时抛出带阶段信息的 `ValueError`，保留旧状态。方差使用 Welford/Chan 的中心矩，对舍入导致的微小负值截零，显著负值视为计算异常。极大有限输入若超出 float64 可计算范围，显式报错。
- 节点中候选特征逐个处理，先保存每列最大增益，再重算第一个满足平局容差的特征，选择最小合格阈值。单节点 `n` 行、`k` 类、`m` 个候选特征的时间约为 `O(m(n log n + nk))`，计数工作区约 `O(nk)`。建树用显式栈执行递归划分，逐特征处理及按节点批量预测路由保留循环。节点数据和形状依赖上一层划分，逐特征还可避免同时分配所有候选的计数数组。CSV、chunk 调度、类别映射、窗口驱逐与绘图允许必要循环；矩、掩码、计数、比较等数组运算使用 NumPy。

## 4. 统计、指标和内存

- **均值、方差、分位数。** 按列忽略 NaN，`RunningStats` 的持久矩状态为 `O(d)`，`count <= ddof` 或全缺失列返回 NaN。分位数初版精确，按块保留已看数值矩阵，在查询时调用旧 `quantile(axis=0, method="linear", nan_policy="omit")`。历史 `N` 行占 `O(Nd)`，查询最坏 `O(dN log N)`，拼接还有 `O(Nd)` 临时内存。Stats 是独立组件，由 demo 选用，Scaler 不保存分位数历史。
- **直方图。** 对每列用相同固定边缘累加计数，`O(db)` 持久存储；末箱含右端点。范围外分别计 underflow/overflow，NaN 不计。固定边缘使跨批计数可加，且便于与离线参考核对。初版提供累计，sliding histogram 为 specs 中的可选项。
- **累计与滚动分类指标。** Trainer 配置固定标签表，训练树在每个缓冲区可只见其中部分标签。固定表用于混淆矩阵及日志；未在树中出现的已配置类别以零概率列对齐，表外标签在 Trainer 学习前报错。每批混淆矩阵加到 `int64` 总矩阵，accuracy 为总对角和除以总评估行数，从行/列总数提取 precision、recall 和 F1。默认零分母用 0，尚无评估行时 accuracy 为 NaN。滚动窗口按**样本**保存最近 `W` 条真值、预测和可用概率，驱逐可切开 chunk；对子矩阵加新计数、减驱逐计数。训练 warmup 不占窗口。
- **AUC。** 有概率的评估行按评估当时的值保存，树更新后不回算。精确累计 AUC 用旧二分类 kernel 计算一对多，平局计半；rolling AUC 对 window 内有概率的行查询。只有正类或只有负类（包括空集）时该类 AUC 为 NaN，宏平均只含已定义的类别，无定义类别则为 NaN。单一真实类别的窗口所有 OvR AUC 都为 NaN。基于 `M` 行概率、`k` 类的累计存储为 `O(Mk)`；查询单个类别 `O(M log M)`，全类别 `O(kM log M)`，窗口以 `W` 为上界。查全量曲线反复排序的成本应与训练计时分开报告。缺概率的行仍进入计数和窗口，并从 AUC 样本中排除；结果返回概率行数供解释。多模型版本概率混合的 AUC 表示整个预序列预测历史。
- **内存。** Trainer 每批成功后记录当前拥有的 ndarray buffer 字节数，按底层存储去重，分列报告树/预处理快照、学习状态、待训样本和指标历史；指向同一数组的引用只计一次，object 等间接对象及 Python 节点、容器、日志对象、解释器、绘图和临时工作区不在此口径。单独的 StreamingStats 历史由调用方持有，可单列其数组估计。该值是批末的实现内存估计，训练期间峰值和进程 RSS 需另行定义与实测；跨 buffer 比较使用同一口径。

## 5. Benchmark、demo 与实施顺序

**循环与向量化。** 复用旧 harness，新写独立纯 Python Gini 候选循环，以相同节点数据、标签、dtype、阈值集合和决胜规则比较生产 kernel。两者接收相同已准备的节点输入，均包含排序与候选计分和选择；数据生成、正确性检查、绘图和 I/O 放在计时外。先对照独立手算/循环和完整结果（候选分数容差、分裂特征、阈值及左右行），再用同一 warmup、重复次数和交替顺序测 `perf_counter_ns`，记录中位数、离散程度、每次样本及环境。

**Buffer sweep。** 多个 `buffer_size` 使用同一不可变数据和 chunk 边界、预处理和树配置、种子及预序列协议。所有运行学习完整流，用各运行都已能预测的共同批次比较主要 accuracy 和 AUC，并报告各自从何时开始预测、覆盖率、全可评估段 accuracy 及各批指标。分开计（1）重训路径，从拼接缓冲、冻结快照及全缓冲变换到建树发布，（2）每批学习路径，（3）推理与评估；磁盘、绘图和日志序列化在外。记录重训次数、实际每次训练行数、单次重训时间分布和累计时间、最后余量及批末内存估计。零次重训时重训时长分位数记缺失；没有共同评估批次就报告缺失并调整实验流长度。用同一环境做独立重复，发布实测 JSON 和可重现脚本，结果注明 seed、数据来源/生成参数、平台、依赖与代码版本。测试不对速度设阈值。

**Demo。** `demo/stream_demo.ipynb` 从 `numcompute_stream.io.iter_supervised_csv` 读入 CSV 并按 chunk 模拟有序数据流，预先声明标签词表。配置 Pipeline 与 StreamTrainer；循环调用 `fit_chunk`，其内部调用 `pipeline.partial_fit`。可独立用 `update_stats` 展示跨批统计。呈现 warmup、重训、累计与滚动记录、流尾余量，再用 `visualise.py` 绘制时间序列和预测对照；预测对照仅收集学习前的真值与预测。预留可运行的 headless 验证路径及交互显示。

| 顺序 | 实现与验证 |
| --- | --- |
| 1 | **已完成。** 建立包入口，实现本阶段需要的列索引、数值及维度验证适配和 CSV 包装，并验证 T01–T03。模型状态和重置契约在对应阶段实现。 |
| 2 | 实现独立 Gini 树和共用缓冲调度，先验证叶概率、分裂、随机性、精确达标和整批越线。 |
| 3 | 复用 `RunningStats` 实现增量预处理和独立 StreamingStats；验证快照、NaN、统计与列顺序。 |
| 4 | 在树及预处理之上接 Pipeline 的原始缓冲与成对发布，验证两次重训之间的坐标。 |
| 5 | 实现指标累计、滚动和 AUC；在 Pipeline 和指标上接 StreamTrainer，验证无泄漏时序、失败重试与日志。 |
| 6 | 接绘图与 demo，新增两个 benchmark 入口并实际测量；运行旧套件与 2.2 测试，验证安装。 |
| 7 | 更新 README，完成最多 4 页 PDF 技术报告及最多 5 分钟视频；在个人 fork 的个人分支整理提交并核对默认分支保留原始小组代码。 |

### 第 1 阶段完成记录

新增 `numcompute_stream/__init__.py`、`_validation.py`、`io.py` 和 `tests/test_stream_io.py`。包入口导出两个旧读取器及新的 `iter_supervised_csv`。适配层复用旧 `integer` 和 `real_array`，补目标列负索引和带输入名的验证错误。

在 `aiml` 环境实测旧读取器会跳过空行、保留尾批，并分别由 NumPy 和旧迭代器检查批内及批间列数。空标签在 `missing="propagate"` 下产生 NaN，文本字段会触发旧解析器的 `ValueError`。新包装层在产出前拒绝 NaN/Inf 标签、Inf 特征及缺少特征列的文件。实现遵循原计划，并明确了两组数组沿用旧读取器的 float64 输出。用现有 setuptools 的 discovery 验证当前规则可找到两个包，wheel 验证留在第 6 阶段。

在仓库根目录用现有 `aiml`（Python 3.12.13、NumPy 2.5.2、pytest 9.2.0）执行如下命令。

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/aiml/bin/python -B -m unittest discover -s tests -p test_stream_io.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/aiml/bin/python -B -m pytest -p no:cacheprovider tests --ignore=tests/test_stream_io.py
```

T01–T03 共 **3 项通过**（0.012 s）。旧套件 **642 项通过，0 项失败，4 条原有 NumPy 配置输出警告**（0.59 s）。两条命令退出码均为 0。第 2–7 阶段待实施。

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
