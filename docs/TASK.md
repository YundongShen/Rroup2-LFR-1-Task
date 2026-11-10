# 任务：复现方法一、二，诊断方法二分类收益有限的原因，实现并评估"均衡 caption→类别桥接"

你在仓库 Rroup2-LFR-1-Task 的分支 `shipeng/bridge` 上工作（已从 main 建好；`scripts/diagnose_bias.py` 已放入）。这是一个课程作业：在标签稀缺的条件下，把 CLIP 适配到遥感数据集 RSICD。请严格按阶段推进，每个阶段结束时停下来向我汇报，得到确认后再继续。进度随时记录在 `docs/PROGRESS.md`（见第九节）。

## 【待填写】
- B 的类别标签能否用于留出 fold 的验证与超参数选择（不用于训练）：**是**。B 的类别标签只能在留出 fold 上用于评估、诊断与超参数选择，绝不作为训练信号（理由见 `docs/bridge_protocol.md` 第 1 节）。凡是以标签指标做选择的地方，都同时报告 mean R@1。
- 计算资源：Slurm 集群，账户 `naiss2026-1-47-gpu`，分区 `gpu`，每个作业 1 块 GPU；本项目可用的 GPU 小时预算：未定总额，每个训练作业控制在 2 小时以内。

## 一、背景与规则

**数据规则（必须严格遵守）：**
- Split A（8,734 张）：只能使用图像。禁止读取 A 的 caption、类别标签，以及文件名（文件名中含有类名，如 `airport_1.jpg`）。
- Split B（1,094 张）：图像与 5 条 caption 可用于训练；类别标签不得用于训练，能否用于验证见待填写项。
- Split C（1,093 张）：只用于最终评估，不得用于训练、模型选择或超参数选择。仓库已有 C 锁（`eval.split=test` 需要 `eval.confirm_test=true`，每次访问都记入 `runs/test_access.log`），不得绕过。
- caption bank 只能由当前 fold 的 **B 训练部分**构成，不得包含留出部分的 caption。

**仓库注意事项：**
- `configs/config.yaml` 的默认值是 `model: vit_l14_laion2b` 和 `device: mps`。每次运行都必须显式指定 `model=vit_b32_laion2b device=cuda`。
- 评估函数 `src/evaluate.py` 直接调用 `model.encode_image`，所以训练好的 adapter 必须包装进 `model.encode_image`，评估时才会生效。
- 新方法要按 `src/methods/__init__.py` 中的接口实现并注册。
**集群环境（详见 CLAUDE.md 的 Running 部分）：**
- 你运行在登录节点上，不得在登录节点运行 GPU 任务。所有 GPU 工作都通过 `sbatch scripts/slurm/run.sbatch <命令>` 提交，在仓库根目录执行。
- Python 环境是 Apptainer 镜像 `/nobackup/proj/disk/cvls/personal/shipengl/rsicd/rsicd.sif`（定义文件 `containers/rsicd.def`）。需要新的 Python 包时，不要在容器外 pip 安装；修改定义文件并告诉我，由我重建镜像。
- 模型权重缓存在 `HF_HOME=/nobackup/proj/disk/cvls/personal/shipengl/rsicd/hf_cache`，已由我预先下载。计算节点可能无法联网，不要依赖运行时下载。
- 写给集群的脚本、作业数组与资源设置都提交到 `scripts/slurm/`，使实验可以被他人重跑。
- `.gitignore` 已排除 `data/raw/`、`runs/`、`*.pt`，不要试图提交这些内容。
- 分支 `yundong/a-only` 上的 `src/ops.py` 有 Sinkhorn 实现（可用 `git show origin/yundong/a-only:src/ops.py` 查看），`docs/a_only_rose.md` 记录了 ROSE 的结论：零样本 CLIP 有严重类别偏置（从不预测 center，过度预测 stadium）；均衡分配带来 +6.9 个点；软标签加每轮重标注会塌缩。不要合并该分支。

**研究问题：** README 中方法二相对方法一，分类只提高 0.91 个点。原因是 caption 监督中的类别信息没有转化为分类能力，还是类别偏置没有被纠正，或是近义类混淆？针对诊断结果提出并验证改进方法。

## 二、阶段 0：环境检查

1. 通过 srun 在计算节点上检查（登录节点是 x86，无法运行 arm64 镜像） `data/splits/` 下三个 CSV 中的所有图像路径是否存在。如有缺失，停下来告诉我。
2. 提交 `sbatch -J zs_kfold5 scripts/slurm/run.sbatch python -m src.run method=zeroshot model=vit_b32_laion2b device=cuda data=kfold5`，作业完成后从日志和 `runs/` 中读取 B 留出 fold 上的结果，作为后续对照。
3. 确认作业日志中没有联网下载权重的记录，GPU 被正确识别。

## 三、阶段 1：复现方法一和方法二

代码不在仓库中，请根据 main 分支 README 的 Method 部分实现。结果不要求与 README 完全一致，大致接近即可。

**共同设定：** OpenCLIP ViT-B/32 `laion2b_s34b_b79k`，CLIP 的图像塔和文本塔全部冻结。唯一可训练的模块是图像侧残差 adapter：$\tilde z = z + \alpha\,h(z)$，$\alpha = 0.2$，$z$ 为冻结 CLIP 的图像特征。

**方法二（`method=semi`）：** $\mathcal L = \mathcal L_{con} + \lambda_{pcap}\mathcal L_{pcap}$。
- 对比损失 $\mathcal L_{con}$（B 训练部分）：对称 InfoNCE，logits $\ell_{ij} = s\cos(\tilde z_i, t_j)$，$s$ 为 CLIP 自带的 logit scale；每张图每步随机取 1 条 caption；如果 batch 中某条 caption 也属于图像 $i$（RSICD 中存在重复的 caption 文本），则它不作为 $i$ 的负例。
- caption bank：当前 fold B 训练部分的所有 caption，按文本去重；$src(j)$ 为拥有该 caption 文本的 B 训练图像集合。
- 每个 epoch 开始时，由 adapter 的 EMA teacher（不是学生）计算：$S_{IT}(u,j)=\cos(z_u, t_j)$，$S_{II}(u,j)=\max_{i\in src(j)}\cos(z_u, z_i)$，其中图像特征都经过 teacher adapter。
- 每张 A 图像各自对整个 bank 做 z-score 后融合：$R = 0.2\,z(S_{IT}) + 0.8\,z(S_{II})$。
- 取 top-3 caption 作为候选，权重 $w_{uj} = \mathrm{softmax}(R/0.1)$（只在候选内归一化）。
- 只保留 $\max_j R(u,j)$ 最高的 50% A 图像（coverage = 0.5）。
- 伪 caption 损失：对被保留的图像，在整个 bank 上计算软交叉熵 $-\sum_{j\in\mathcal N_u} w_{uj}\log\mathrm{softmax}_{j'}(s\cos(\tilde z_u, t_{j'}))$。
- $\lambda_{pcap}$ 在前 20% 的训练步数中从 0 线性升到 1。

**方法一：** 不单独实现。在方法二的代码中加一个开关 `pcap.enabled`，关闭时不计算伪 caption 损失、不使用 A，其余（adapter、对比损失、优化器、训练步数）与方法二完全相同。方法一即 `method=semi pcap.enabled=false`。

**README 没有写明的细节，使用以下默认值，并全部记录在 `docs/reproduction.md` 中：**
- $h$：两层 MLP，512 → 512 → 512，GELU 激活，最后一层零初始化（训练开始时 adapter 等于恒等映射）。
- 优化器：AdamW，lr = 1e-3，weight decay = 1e-4，余弦学习率衰减。
- EMA 动量：0.999（每步更新）。
- batch：每步一个 B batch（128 对）加一个 A batch（128 张被保留的图像）；一个 epoch 定义为遍历一遍被保留的 A 图像，B 循环采样。方法一使用相同的总步数。
- 训练长度：先用 fold 0 观察 B 留出部分的 mean R@1 曲线，取一个足以收敛的步数，然后固定下来，所有方法、fold、种子都用同一个值。
- 因为 CLIP 是冻结的，可以预先缓存所有图像特征和 caption 文本特征，训练只在特征上进行。但最终评估必须通过包装后的 `model.encode_image` 走仓库的标准 `evaluate` 流程，并验证缓存特征路径与标准路径的结果一致。

**验收标准**（只用 B 留出 fold，不访问 C）：
- 在 `data=kfold5` 上，种子 0、1、2。
- 方法二的 mean R@1（两个检索方向 R@1 的平均）与 README 参数扫描表中 coverage 0.5、top-3 的 22.53 ± 0.69 相差不超过约 2 个点。
- 趋势与 README 一致：方法一、方法二的分类准确率都明显高于 zero-shot；方法二的检索指标高于方法一。
- 达不到时，只允许针对方法二在 B 留出 fold 上调整 lr、训练长度和 MLP 宽度（以 mean R@1 为选择指标），选出的设定同样用于方法一，并记录所有尝试。仍然达不到时，停下来向我汇报，不要继续。

产出 `docs/reproduction.md`：实现说明、自选的默认值、B 上的结果表（与 README 对照）、与 README 不一致之处。

## 四、阶段 2：诊断

1. 在 `src/run.py` 中，每个 fold 执行 `evaluate(...)` 后，如果 `cfg.eval.split == "val"`，就用 `src.evaluate.encode_images` 对 `b_val` 重新编码，并调用 `scripts/diagnose_bias.py` 中的 `save_val_logits`，保存到 `<run_dir>/val_logits_fold{k}.pt`。类别嵌入用该脚本的 `class_text_embeddings`，模板为 "A satellite image of {}."。
2. 先检查诊断脚本与仓库的 fold 划分、prompt、数据读取是否一致；为其中的纯 numpy 函数写单元测试（放在 `tests/`），包括 Sinkhorn 的行和与列和、均衡分配能否恢复人为注入的偏置。
3. 对方法一、方法二的种子 0、1、2 分别运行诊断，汇总均值与标准差。
4. 按以下**预先设定的判据**下结论（不得在看到结果后修改）：
   - 支持偏置假设：方法二的 `acc_after_balancing_uniform` 比 `acc` 高 2 个点以上，并且存在从不被预测或严重低估（预测占比低于真实占比的一半）的类。
   - 支持转化假设：`pseudo_captions/caption_right_when_image_wrong` ≥ 0.30，并且 `pseudo_captions/acc_image_x_caption` 比 `acc` 高 1 个点以上。
   - 支持近义类假设：`err_share_into_3_text_nearest` ≥ 随机水平的 2 倍。
   - 比较方法一和方法二的偏置指标，说明 caption 监督是否减轻了偏置。
5. 产出 `docs/diagnosis.md`：各假设的指标表与结论、逐类表（重点标出 center、stadium、playfields、sparse residential、commercial 等类）、预测分布与真实分布的对比图。

## 五、阶段 3：实现"均衡 caption→类别桥接"

在方法二的基础上新增一个损失。方法二原有部分全部保留，可训练参数仍然只有 adapter。所有新逻辑通过配置开关控制；关闭时，结果必须与方法二逐位一致（用固定种子验证）。

**训练前的一次性准备**
1. 类别文本嵌入 $v_c$：用冻结文本塔编码 31 个类名。训练模板用 "an aerial photo of {}."，与评估模板 "A satellite image of {}." 区分开。
2. caption→类别矩阵：$a_{jc}=\mathrm{softmax}_c(\cos(g(caption_j), v_c)/\tau_{text})$，对 caption bank 计算一次并缓存。
3. 先验 $\pi$：`uniform`（1/31），或 `caption`（当前 fold B 训练部分所有 caption 的 $a_{jc}$ 平均），或两者按 $\gamma$ 插值。不得使用任何类别标签。

**每个 epoch 开始时**（在 teacher 完成伪 caption 分配之后）
1. 对**全部** A 图像计算 $r_{uc}=\sum_j q_{uj}a_{jc}$，$q_{uj}$ 为方法二的 top-3 caption 权重。
2. 在**全部** A 图像上做非平衡 Sinkhorn：输入为 $\log r / \varepsilon_b$；每行和为 1；类别总量每轮按 $Q \mathrel{*}= (\pi / \text{当前类别总量})^{\kappa}$ 部分拉向 $\pi$。$\kappa=1$ 为完全均衡，$\kappa=0$ 为不均衡。
3. 硬标签 $\hat y_u = \arg\max_c Q_{uc}$。
4. 只对方法二已保留的图像 $\mathcal U_{sel}$ 计算损失。
5. 记录：$r$ 的平均熵、伪标签的类别直方图、与上一个 epoch 相比改变的标签比例。

**每个 batch**
- $p_{uc}=\mathrm{softmax}_c(s\cdot\cos(\tilde z_u, v_c))$。
- $\mathcal L_{bridge} = -\mathrm{mean}_{u\in\mathcal U_{sel}}\log p_{u\hat y_u}$。
- $\mathcal L = \mathcal L_{con} + \lambda_{pcap}\mathcal L_{pcap} + \lambda_{bridge}\mathcal L_{bridge}$，$\lambda_{bridge}$ 在前 20% 训练中从 0 线性升到目标值。

**配置项：** `bridge.enabled`；`bridge.source`（`caption` 为主方法，`image` 为对照：用 teacher 的图像–类名相似度代替 $r$，其余完全相同）；`bridge.prior`（`uniform` / `caption` / `mix`）与 `bridge.gamma`；`bridge.kappa`；`bridge.eps_b`；`bridge.tau_text`；`bridge.lambda`；`bridge.target`（`hard` / `soft`）。

为 Sinkhorn（含 $\kappa$ 与先验）和 $r$ 的计算写单元测试。

## 六、阶段 4：实验

**选参规则**（运行前写进 `docs/bridge_protocol.md`）：
- B 标签可用于验证时：在 B 五折上以分类准确率为选择指标，约束为 mean R@1 相对方法二下降不超过 0.5 个点。
- 不可用时：不做选择，直接用默认值 $\tau_{text}=0.01$、$\varepsilon_b=1$、$\kappa=1$、`prior=uniform`、$\lambda_{bridge}=0.5$，并在报告中说明。

**网格**（先用种子 0 粗筛，再对前 3 名用种子 0–2 复核）：$\lambda_{bridge}\in\{0.25,0.5,1.0\}$；$\kappa\in\{0,0.5,1\}$；`prior` ∈ {uniform, caption}；$\varepsilon_b\in\{0.5,1.0\}$。如果 $r$ 的平均熵过高（接近均匀）或过低（接近 one-hot），先调整 $\tau_{text}$ 并记录理由。

**对照实验**（相同的选参流程）：
1. 方法二
2. 方法二 + 桥接，$\kappa=0$（原始桥接方案，无均衡）
3. 方法二 + 均衡桥接（主方法，最优配置）
4. 方法二 + 均衡桥接，`source=image`（检验 caption 是否提供了图像之外的类别信息）
5. 主方法，`target=soft`（检验塌缩风险）
6. 方法二 + 测试时均衡（不训练，只对评估集的预测做 Sinkhorn）。这是直推式方法，用到了评估集的无标签图像，在所有表格中单独标注。

每个实验记录：分类准确率、每类平均准确率、预测分布与真实分布的 TV 距离、两个方向的 R@1/5/10、逐类准确率。

## 七、阶段 5：最终评估

只有在我确认最终配置之后，才在 C 上运行：零样本、方法一，以及对照 1–5，各跑种子 0–4，报告均值 ± 标准差；对主方法与方法二做配对检验（同一种子配对）。C 上只运行这一次，不得根据 C 的结果回头修改任何配置。

## 八、产出

- `docs/reproduction.md`、`docs/diagnosis.md`、`docs/bridge_protocol.md`（实验前写好）、`docs/bridge_results.md`（B 上的选参结果、C 上的最终结果表（格式与 README 一致）、消融、敏感性图、逐类分析、失败案例）。
- 所有结果表都能由 `scripts/` 下的脚本从 `runs/` 重新生成。

## 九、工作规则

- 维护 `docs/PROGRESS.md`：当前所处阶段、已完成的事项、关键数字、待我决定的问题、下一步。每完成一个步骤就更新并提交。新会话开始时先读它。
- 小步提交，每个提交只做一件事，提交信息写清楚。只推送到 `shipeng/bridge`，不得修改其他分支。
- 新代码先在小规模数据上跑通（可用一个短时限的作业，如 `-t 00:20:00`），再进行完整实验。提交前估算 GPU 小时，单次提交合计超过 10 GPU 小时须先问我。
- 以下情况必须停下来问我：图像缺失；复现未达验收标准；数据使用规则不明确；诊断结论与预期矛盾；任何需要访问 C 的操作；需要偏离本文档的设计。
- 不得为了让结果好看而修改判据、网格或选参规则。负面结果同样如实记录。
- 每个阶段结束时按以下格式汇报：完成了什么；关键数字（附标准差）；发现的问题或异常；下一步计划和需要我决定的事项。