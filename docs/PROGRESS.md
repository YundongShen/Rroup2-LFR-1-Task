# PROGRESS

## 当前阶段
阶段 4（B 上选参与对照实验）已完成，结果见 `docs/bridge_results.md`。等待用户批准后才进入阶段 5（C）。

## 已完成
### 阶段 0（2026-11-10）
1. 图像路径检查（计算节点 n522，容器内，作业 `env_check` 5088328）：
   - A_unlabeled.csv 8734 行，缺失 0，重复 0（只检查存在性，未输出 A 的路径）
   - B_labeled.csv 1094 行，缺失 0，重复 0
   - C_test.csv 1093 行，缺失 0，重复 0（只检查文件存在，未读取标签或 caption）
   - 环境：torch 2.5.0a0 (nv24.10)，open_clip 3.3.0，GPU NVIDIA GH200 120GB，`cuda=True`
2. 零样本基线，`data=kfold5`，在 B 留出 fold 上评估（作业 `zs_kfold5` 5088327，耗时 35 s）
   - 运行目录：`runs/zeroshot_vit_b32_laion2b/2026-11-10_00-21-55`
   - fold 大小：train 875/875/875/875/876，val 219/219/219/219/218
3. 日志检查：权重从 `hf_cache` 本地快照加载（文件时间 2026-11-09 22:45，早于作业），没有下载记录。
   但 huggingface_hub 向 huggingface.co 发了一次 HEAD 请求（返回 302），说明计算节点能联网，
   代码也会在运行时访问网络。

### 阶段 1（2026-11-10）
详见 `docs/reproduction.md`。
1. 代码：`src/features.py`（冻结特征缓存），`src/methods/semi.py`（方法二，`pcap.enabled=false` 即方法一），
   `configs/method/semi.yaml`，`tests/test_semi.py`（5 项通过），`run.py` 增加 `folds` 子集与缓存/标准路径对比。
2. 训练长度：fold 0 曲线（预先登记的规则）选出 T = 500。更长训练过拟合（T=8000 时 mean R@1 18.0）。
3. 正式运行：kfold5 × 种子 0–2，作业 `p1_main` 5127397（2 分 48 秒）。
4. 缓存路径与标准路径：40 个 fold 中 35 个完全一致，5 个有一张图的 I→T 排名差一位（重复 caption 的精确平局）。

### 阶段 2（2026-11-10）
详见 `docs/diagnosis.md`。
1. `run.py` 在 `eval.split=val` 时保存 `val_logits_fold{k}.pt`；重跑方法一、二种子 0–2（与阶段 1 逐位一致）。
2. 核对 `scripts/diagnose_bias.py` 与仓库（fold 划分、prompt、数据、伪 caption 打分）一致；
   `tests/test_diagnose_bias.py` 11 项，全部 16 项测试通过。
3. 诊断（作业 p2_diag 5130298）与汇总（`scripts/diagnosis_report.py`，含混淆对与分布图 `docs/figs/pred_vs_true.png`）。

### 阶段 3（2026-11-10）
用户选择 A：照原设计实现并如实评估。
1. `src/methods/semi.py`：`caption_class_matrix`、`class_prior`（uniform / caption / mix = (1−γ)·uniform + γ·caption）、
   `bridge_r`、缩放向量形式的 `unbalanced_sinkhorn`（对数域、float64，κ 作用在每轮重算的 β 上）、`bridge_loss`
   （hard / soft）；`source=image` 用 teacher 图像–类名相似度 softmax(cos/τ_text) 代替 r。配置 `method.bridge.*`，默认关闭。
2. `tests/test_bridge.py`：κ=0 等于逐行归一化 r^{1/ε}；κ=1 列和 = Nπ（均匀与非均匀先验，ε=0.5/1）；κ=0.5 列和介于两者之间；
   κ 不累乘（收敛后仍不均衡）；先验、r、损失。全部 29 项测试通过。
3. 关闭桥接 = 方法二逐位一致（种子 0、5 折：指标与 adapter 权重完全相同；作业 p3_smoke 5131061）。
4. 冒烟（种子 0，fold 0，T=500）：默认配置 H(r) ≈ 0.48 nats（最大 3.43），Q 列和与先验一致，A 上硬标签每类 232–350 张，
   每轮改变 0.4–0.8 %。κ=0 时 A 上硬标签每类 0–993 张（r 的类别总量与均匀 TV 23.7 %）；prior=caption 时 0–868 张。
   fold 0 数值仅作冒烟，不作结论。

### 阶段 4（2026-11-10）
详见 `docs/bridge_protocol.md`（先提交 bbafdf3）与 `docs/bridge_results.md`。
1. 伪标签质量（用户追加，仅评估，`scripts/pseudo_label_quality.py`）：caption 来源最好的是 κ=1、uniform，79.5 %（EMA teacher）。
   所有设置都低于方法二自身的分类准确率 83.4 %。κ=0 时 playfields 在留出 B 与 A 上都得到 0 个硬伪标签（r 份额 0.01 %）。
2. 粗筛网格：种子 0，54 个配置（作业 p4_grid 5132538，31 分钟）。κ=0 时硬目标与 ε_b 无关，ε 0.5/1 两组完全相同，
   caption 网格实际只有 27 个不同配置（协议写的是 30 个）。规则未改。
3. 复核：每组前 3 名用种子 1、2 复核，并加上 #5 soft（作业 p4_recheck 5133405，14 分钟）。最后一步因 `sys.path` 失败，
   修复后单独重跑 `bridge_results.py final`（srun）。分析脚本 `scripts/bridge_analysis.py` 输出敏感性图、混淆对和失败案例。
4. 选中配置：#2 caption_l0.25_e1.0_k0；#3（主方法）caption_l0.25_e1.0_k0.5_uniform；#4 image_l0.25_e0.5_k1_caption；
   #6 κ 1、ε_b 1、uniform。

## 关键数字：阶段 4，B 留出 fold（种子 0–2，均值 ± 标准差，%）
| 实验 | acc | 每类平均 | TV | mR1 |
|---|---|---|---|---|
| #1 方法二 | 83.43 ± 0.23 | 84.57 ± 0.24 | 11.52 | 23.84 ± 0.12 |
| #2 桥接 κ=0 | 82.24 ± 0.31 | 83.83 ± 0.32 | 13.65 | 23.90 ± 0.27 |
| #3 均衡桥接（主方法） | 83.88 ± 0.38 | 85.61 ± 0.39 | 12.43 | 24.09 ± 0.12 |
| #4 source=image | 81.11 ± 0.11 | 82.62 ± 0.09 | 15.33 | 23.90 ± 0.22 |
| #5 主方法 soft | 83.37 ± 0.22 | 85.16 ± 0.21 | 13.01 | 24.01 ± 0.20 |
| #6 测试时均衡 †（转导式） | 85.80 ± 0.22 | 87.50 ± 0.20 | 7.04 | 23.84 ± 0.12 |

主方法比方法二高 +0.45，约为 1.2 倍种子标准差，且是在同一批 fold 上从 24 个配置中选出的。不能算确认的提升。
playfields 从 15.9 降到 1.0。

## 关键数字：阶段 2，方法二（3 个种子均值；预先设定的判据）
| 假设 | 指标 | 值 | 结论 |
|---|---|---|---|
| H1 偏置 | 均衡后 acc − acc（需 > 2）；低估类数 | +0.70（1.01/0.64/0.46）；3 | 不支持 |
| H2 转化 | caption_right_when_image_wrong（需 ≥ 30 %）；acc_image_x_caption − acc（需 > 1） | 30.70 %；+0.98（1.19/0.73/1.01） | 不支持（临界） |
| H3 近义类 | err_share_into_3_text_nearest / 随机水平（需 ≥ 2） | 53.1 % / 10 % = 5.3× | 支持 |

- 主要错误：playfields → playground（占全部错误 21 %），sparse residential → medium residential（13 %）。
  playfields 召回 15.9 %（方法一 0 %），sparse residential 0 %，center 17.9 %（方法一 41.0 %）。
- 这些类的 caption→类别映射本身就错：用图像自己的人工 caption，playfields 召回 0 %，sparse residential 16.7 %，
  center 3.8 %（B 中 playfields 的 335 条 caption 有 135 条含 "playground"，无一含 "playfield"）。
- 方法一 → 方法二：TV 12.22 → 11.52 %，过度预测类 1 → 0.33，低估类 2.3 → 3.0，从不预测 0 → 1；偏置只略有减轻。
- caption 先验离真实分布远（TV 21 %），均匀先验 9 %。

## 关键数字：阶段 1，B 留出 fold（每个种子取 5 折均值，再报告 3 个种子的均值 ± 标准差，%）
| 方法 | acc | per-class acc | mean R@1 | I→T R@5 | T→I R@5 |
|---|---|---|---|---|---|
| 零样本 | 63.62 | 63.50 | 13.76 | 31.08 | 49.25 |
| 方法一 | 82.97 ± 0.31 | 84.66 ± 0.33 | 22.72 ± 0.12 | 51.80 ± 0.86 | 68.33 ± 0.30 |
| 方法二 | 83.43 ± 0.23 | 84.59 ± 0.25 | 23.84 ± 0.12 | 49.67 ± 0.55 | 67.90 ± 0.09 |

验收：mean R@1 与 README 22.53 相差 +1.31（通过）；分类明显高于零样本（通过）；
方法二 mean R@1 高于方法一（通过），但两个方向的 R@5 都低于方法一（与 README 测试表趋势不一致，待用户判断）。

## 关键数字：零样本，B 留出 fold（5 折均值 ± 折间标准差，种子无关）
| 指标 | 值 (%) |
|---|---|
| 分类准确率 zs_acc | 63.62 ± 1.29 |
| 每类平均准确率 | 63.50 ± 1.22 |
| i2t R@1 / R@5 / R@10 | 13.71 ± 1.58 / 31.08 ± 2.30 / 46.16 ± 4.26 |
| t2i R@1 / R@5 / R@10 | 13.80 ± 1.29 / 49.25 ± 2.54 / 66.14 ± 2.55 |
| mean R@1 | 13.76 ± 1.02 |

逐折 mean R@1：14.34 / 12.33 / 12.74 / 14.61 / 14.77。
（README 中 C 上的零样本准确率为 63.76，与 B 上的 63.62 相近。）

## 用户决定（2026-11-10）
- B 类别标签：只可在留出 fold 上用于评估、诊断与选参，绝不作为训练信号；凡用标签指标做选择时
  同时报告 mean R@1。已写入 TASK.md【待填写】和 `docs/bridge_protocol.md` 第 1 节。阶段 1、2、4 按原文执行。
- GPU 预算：未定总额，每个训练作业控制在 2 小时以内。
- 本地 `run.sbatch` 加 `HF_HUB_OFFLINE=1`（作业 5123960 验证：无网络请求，零样本结果与第一次逐位一致）。

- TASK.md 更新（用户）：集群文件（`scripts/slurm/`、`containers/`）只留本地，不得提交；只提交方法代码、
  Python 分析脚本、测试与文档。已恢复 `.git/info/exclude` 中的 `scripts/slurm/`，并用 filter-branch 把它从
  本地未推送的历史中移除（旧 HEAD 79a5430 / 398075c 保存在 `refs/original/` 与 reflog）。作业命令改为写进文档。
- TASK.md 更新（用户）：阶段 3 的非平衡 Sinkhorn 改为缩放向量形式（κ 作用在每轮重算的 β 上），并规定了三项单元测试。

## 推送
- 不由我推送：所有提交留在本地，等整个方案完成后由用户统一 push。

- 阶段 1 验收（用户）：按 mean R@1 判通过；R@5 偏低记录在案。T 保持 500（用户允许我自行修改；
  更长训练只会过拟合，提高 teacher 作用需改 TASK 规定的 EMA 动量，故不改）。
- 用户已自行提交并推送（23c776e 等），README 改写为桥接方法报告框架。

## 待用户决定
- 阶段 4 完成：是否批准阶段 5（C，种子 0–4），以及在 C 上跑哪些配置（见 `runs/p4/chosen.json`）。

## 下一步
阶段 5，需用户批准后才开始：在 C 上评估批准的配置（种子 0–4），结果表格式与 README 一致。
