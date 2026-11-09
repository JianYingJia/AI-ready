# AI-ready：面向 Benchmark 的文档数据估值

面向“多个候选文档批次 → 小预算代理试训 → 预测高预算训练价值”的研究代码。系统保留已有的可追溯文档底座，并新增批次设计、客观语义单元挖掘、覆盖与质量保持型 token 预算选点、高预算 Production Contract、全遍历审计、配对种子效用和排序／后悔值评估。

```text
候选批次 D1...Dn + 用户 Benchmark 能力蓝图
  → 每批独立解析、清洗、去重和来源登记
  → 只从来源文本挖掘规则、条件、例外和时效语义单元
  ├→ 覆盖与质量分布保持型代理选点 → 仅对入选单元生成小样 SFT
  └→ Fully Open Meditron 完整处理同批原始文档 → 全量参考 SFT
  → 两条独立支路使用相同 LoRA-SFT 与 Benchmark 评测条件
  → Spearman / Kendall / Top-k / Top-1 regret / 排名反转
```

原 `Topic × Task × Difficulty` benchmark 构造代码暂时保留为兼容支线，不再代表当前数据估值主张。数据估值主线不会读取锁定测试题，只读取不含题文和答案的能力蓝图；测试题仅由评测适配器访问。

## 当前可运行实验：CLL/SLL

当前工作区已完成一套 **CLL/SLL（慢性淋巴细胞白血病／小淋巴细胞淋巴瘤）公开可行性数据**的获取与冻结。该实验用于回答一个具体问题：能否从每个候选数据批次中选出一个低预算“小样”，让小样试训后的跨批次排序接近全批次高预算训练的排序。

主实验采用同一基础模型和同一套 LoRA-SFT 配方，但两臂的数据生成相互独立：低预算组先用覆盖方法选择原始知识单元，再只对入选单元生成小样 SFT；高预算组由 Fully Open Meditron 完整处理同批原始文档，生成全量参考 SFT。小样不是从 Fully Open Meditron 的全量产物中抽取。CPT 可以作为后续敏感性分析，但不作为本轮方法筛选的主实验。最终依据是两条支路的批次排序、Top-k 命中和 Top-1 regret 是否一致，并与 coverage-only、分层随机、随机、相关性和多样性选点基线比较。

因此，本轮主实验检验的是“覆盖小样流水线能否预测 Fully Open Meditron 全量流水线的批次训练价值”。两条支路的生成器不同，主实验本身不能把排序一致性完全归因于覆盖选择器；若要单独估计选择机制的作用，需要另设同生成器消融实验。

| 项目 | 当前状态 |
| --- | --- |
| 数据快照日期 | `2026-09-13` |
| 公开医学来源 | 6 个 NCI／PMC 文本来源 |
| 候选批次 | 12 个：8 个受控批次＋4 个自然检查批次 |
| 代理 Benchmark | 46 道锁定的 MedMCQA CLL/SLL 题目 |
| 公开数据包 | `data/acquired/valuation_cll/`，103 个文件，约 2.82 MiB |
| 实际候选批次 | `data/acquired/valuation_cll/batches/`，81 个文件，约 0.62 MiB |
| Benchmark 上游快照 | `data/private/valuation_cll/upstream/`，约 81.92 MiB |
| 锁定 Benchmark 题目文件 SHA-256 | `4382db1bb0884de73f34d7144876b41e06c653452af74d51d812ec9827fc8ae5` |
| Benchmark manifest SHA-256 | `36ada151de2468e79bbcb52086ebac71e92fb95150995100dc58d7dd1656724d` |
| 冻结实验配置 | `configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml` |
| 冻结可行性协议 | `configs/valuation_cll_public_feasibility/feasibility_protocol.yaml` |
| 完整实验矩阵 | 12 批次 × 7 实验臂 × 3 训练种子 = 252 条 trial |
| 训练框架 | LLaMAFactory `v0.9.5`，Qwen2.5-7B-Instruct LoRA-SFT，一卡一 trial |
| 训练端冻结配置 | `configs/valuation_cll_public_feasibility/llamafactory/server_profile.yaml` |
| 历史实验清单 | `outputs/cll_feasibility_plan_20260916_v3/experiment_manifest.json`；其合同哈希属于旧训练栈，不能用于当前训练 |
| 本次交付状态 | 已完成本地试验生成和服务器小样/全量训练验证；结果确认小样能够保持全量训练效用 |

训练数据和评测数据按目录与接口隔离：批次构造、知识单元挖掘和代理选点流程约定只读取 `data/acquired/`；锁定题目位于 `data/private/`，按协议只由评测适配器读取。冻结配置中不含私有路径，获取锁记录来源版本、文件哈希和 Benchmark 哈希。

首次获取或复现当前数据：

```powershell
conda activate aiReady
cd D:\pythonCode\research\AI-ready\ai-ready-benchmark

python tools/acquire_cll_data.py

python tools/value_data.py validate `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml

python tools/value_data.py plan-experiment `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml `
  --protocol configs/valuation_cll_public_feasibility/feasibility_protocol.yaml `
  --output-dir outputs/cll_feasibility_plan_llamafactory
```

获取命令默认复用已有快照。只有建立新数据版本时才使用 `python tools/acquire_cll_data.py --refresh`，并应检查 `acquisition_lock.json` 中发生变化的 SHA-256。当前关键产物如下：

`data/acquired/` 和 `data/private/` 都受 `.gitignore` 管理，不随 Git checkout 分发；在新环境中必须先运行获取命令。首次下载约 86 MB。若只保留锁定题集，可运行 `python tools/acquire_cll_data.py --drop-benchmark-upstream` 删除约 82 MiB 的上游 Parquet；需要完整离线复现筛选过程时应保留该文件。

```text
data/acquired/valuation_cll/
├── raw/                    # NCI HTML 与 PMC JATS XML 原始快照
├── processed/              # 纯文本、结构块与来源信息
├── batches/B01...B12/      # 12 个候选训练批次
├── license_evidence/       # 许可依据快照
└── acquisition_manifest.json

data/private/valuation_cll/
├── upstream/medmcqa-train-91c6572c4540.parquet
└── benchmark_locked/medmcqa_cll_proxy_v1.jsonl

configs/valuation_cll_public_feasibility/
├── acquisition_lock.json
├── capability_blueprint.yaml
├── benchmark_test_manifest.json
├── batch_suite.yaml
├── batch_suite.frozen.yaml
├── feasibility_protocol.yaml
├── llamafactory/server_profile.yaml  # 服务器硬件、batch 与运行时冻结配置
├── proxy_selection/        # 主方法、消融与四类基线的冻结选择配置
├── stage_ablation_protocol.yaml      # 11-study 阶段消融与判定终点
├── stage_ablation_execution.template.yaml  # 32 个同生成器 SFT cell 的执行模板
└── production_contract.yaml
```

这 46 道题是从固定 MedMCQA 公开训练分片一次性划出并锁定的研究 holdout，登记为 `audit_only`，不能进入 SFT。它适合流程联调和方法筛选，但不是 MedMCQA 官方隐藏测试集，也不能代替用户的真实 Benchmark。完整的数据来源、12 批次设计、许可边界和能力分布见 [`docs/CLL数据获取与Benchmark说明.md`](docs/CLL数据获取与Benchmark说明.md)；方案与当前实验的汇报稿见 [`docs/20260915_文档数据估值方案与CLL实验进展汇报.md`](docs/20260915_文档数据估值方案与CLL实验进展汇报.md)。

当前已完成数据、实验协议、两条独立生成支路的来源审计和三态判定代码，并已把训练端绑定到 LLaMAFactory `v0.9.5` 与固定 Qwen2.5 模型 revision。覆盖小样命令现在严格执行“先选择、后生成”；Fully Open Meditron 全量结果通过独立导入命令完成完整遍历审计。此前本地 545 条 SFT 产物属于旧版 `local-pilot/non-formal` 链路，不作为新版双支路正式工件。项目组已确认服务器小样／全量配对训练的阶段性结果；正式复现实验需要按新版契约重新登记两条支路的生成清单，将代理工件精确绑定到 `generation.proxy` 配方，并为每个 Fully Open Meditron 批次保留外部作业的 `fully_open_meditron_run_manifest.json`。该上游运行清单绑定 FOM revision、工具 adapter/repository/config 哈希、teacher revision、prompt/质量门哈希、合同与批次身份、eligible 输入和规范化输出哈希，以及未访问 Benchmark、未使用代理选择器、完整遍历的声明；导入后的 `generation_manifest.json` 会继续绑定这份上游清单。同时还需同步逐 trial 分数和结果哈希。8—12 个批次的结果仍只用于可行性和方法筛选。承接上次汇报的阶段性版本见 [`docs/20260922_从内容覆盖到训练效用覆盖的阶段性进展汇报.md`](docs/20260922_从内容覆盖到训练效用覆盖的阶段性进展汇报.md)，完整实验逻辑见 [`docs/20260922_小样代表全量的实验逻辑与近期进展汇报.md`](docs/20260922_小样代表全量的实验逻辑与近期进展汇报.md)，总体实验步骤见 [`docs/CLL小样可行性实验执行说明.md`](docs/CLL小样可行性实验执行说明.md)，服务器训练命令见 [`docs/LLaMAFactory服务器训练说明.md`](docs/LLaMAFactory服务器训练说明.md)。

## 条件价值画像小样实验

仓库新增“数据属性 × Benchmark 能力 × Base Model 初始掌握状态”的联合分层抽样实验。它用于生成估值小样，不按模型缺口优选数据，也不在这一阶段讨论训练效用。小样由覆盖方法选择语义单元并另行生成；全量参考由 Fully Open Meditron 完整处理同批原始材料。两条生成路径不能混用。

代码同时支持能力级 `capability_prior` 和逐“语义单元 × 能力”的 `unit_cell` 模型画像。现已用冻结的 Qwen2.5-7B-Instruct checkpoint 在 60 道独立 CLL 探针上完成 120 次真实推理，六个能力均达到最低探针数，能力级画像状态为 `complete`。本地运行采用 NF4 量化加载，且粒度仍是能力级先验；它补齐了粗粒度 M 画像，但不等同于 BF16 画像或逐语义单元掌握度。

```powershell
python tools/value_data.py run-model-probes `
  --catalogue inputs/conditional_value/cll_independent_capability_probes_v1.jsonl `
  --config configs/valuation_cll_public_feasibility/conditional_value/base_model_probe.qwen2_5_7b.yaml `
  --output-dir outputs/cll_base_model_profile_qwen2_5_7b_20260926/probe_run_nf4_v2_contract_bound

python tools/value_data.py adapt-value-population `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --artifact-root outputs/cll_local_sft_qwen2_5_1_5b_full_20260921 `
  --output-dir outputs/cll_conditional_value/population

python tools/value_data.py sample-value-proxy `
  --units outputs/cll_conditional_value/population/value_population_units.jsonl `
  --model-profile outputs/cll_conditional_value/model_profile.json `
  --config configs/valuation_cll_public_feasibility/conditional_value/value_sampling.yaml `
  --batch-id B01 `
  --membership-map outputs/cll_conditional_value/population/membership_map.jsonl `
  --artifact-root outputs/cll_local_sft_qwen2_5_1_5b_full_20260921 `
  --output-dir outputs/cll_conditional_value/samples/B01

python tools/value_data.py verify-value-sample-manifest `
  --manifest outputs/cll_conditional_value/samples/B01/value_sample_manifest.json `
  --population-units outputs/cll_conditional_value/samples/B01/annotated_population_units.jsonl

python tools/value_data.py sweep-value-fidelity `
  --units outputs/cll_conditional_value/population/value_population_units.jsonl `
  --model-profile outputs/cll_conditional_value/model_profile.json `
  --config configs/valuation_cll_public_feasibility/conditional_value/value_sweep.yaml `
  --thresholds configs/valuation_cll_public_feasibility/conditional_value/fidelity_thresholds.yaml `
  --critical-cell-weights configs/valuation_cll_public_feasibility/conditional_value/critical_cell_weights.yaml `
  --output outputs/cll_conditional_value/budget_seed_sweep.json
```

抽样命令会同时输出当前 schema v2 的抽样清单、完整带注释总体、带纳入概率和设计权重的入选记录，以及供覆盖生成器使用的语义单元、代理单元和覆盖义务子集。抽样清单可以根据完整总体重放，保真证书还会检查 HT 总量误差和按依赖组计算的有效样本量。

当前可部署分层只使用抽样前已知的 `knowledge_type`、`quality_bucket`、`benchmark_relevance` 和 `source_scope`；生成后的 `generation_status` 只做留出审计。当前六个 Benchmark 能力权重均为 1.0，`rare_stratum_quotas` 为空，二者都还是正式实验前需要补齐的登记项。预算扫描会在多个比例和随机种子下寻找满足画像误差门槛的最小实际比例，避免预设“25% 对所有批次都够用”。

当前能力级 Base Model 画像、固定样本、画像和证书在 `outputs/cll_base_model_profile_qwen2_5_7b_20260926/`，最终批次工件位于其 `v3_contract_bound_capability_profile/` 子目录：12/12 批次通过已测分布门槛，12/12 抽样来源重放和证书重放通过，总状态为 `complete`。这里的 `complete` 只表示六个能力的探针测量完整；逐单元 M 画像仍未完成。完整结果见 [`docs/20260926_真实Base_Model画像补齐结果.md`](docs/20260926_真实Base_Model画像补齐结果.md)，探针复核见 [`docs/20260926_CLL_Base_Model探针人工审查记录.md`](docs/20260926_CLL_Base_Model探针人工审查记录.md)，实验逻辑见 [`docs/20260926_条件价值驱动的覆盖小样生成机制实验汇报.md`](docs/20260926_条件价值驱动的覆盖小样生成机制实验汇报.md)。

## 阶段级消融实验

现有主实验比较覆盖小样流水线与 Fully Open Meditron 全量流水线，回答跨流水线预测问题。为了进一步解释“语义被覆盖后为什么会变成训练价值”，仓库增加了独立的阶段级消融协议，覆盖语义挖掘、蓝图条件化、子集选择、质量保真、监督组织、可学习性、规模／重复／噪声、Base Model 先验和最终 FOM 外部效度。因果消融固定使用同一覆盖生成器；FOM 只保留为生产外部参考，正式 `ProductionContract v2` 的双支路约束没有放宽。

当前冻结协议登记 **11 个 study、35 个 cell 和 35 个判定终点**。其中 32 个同生成器 SFT cell 由消融执行层物化；A1 的 2 个 cell 是不训练的挖掘审计，A9 的 `fom_operational_full` 由独立 Fully Open Meditron 作业提供。

面向汇报的完整实验总表见 [`docs/20260926_完整实验内容_输入输出指标与案例.md`](docs/20260926_完整实验内容_输入输出指标与案例.md)。该文档把前置可辨识性、真实 Base Model 画像、11 个阶段实验、预算与内容覆盖以及 FOM 端到端预测分别成章，逐项给出实验目的、输入、输出案例、指标用途、判定门槛和当前状态。

近期关于“研究问题怎样形式化、小样为什么能够预测全量价值、能力级 Base Model 画像为什么可能使 A8 不可识别”的讨论，集中整理在 [`docs/20260927_代表性小样研究的核心问题_机理与形式化说明.md`](docs/20260927_代表性小样研究的核心问题_机理与形式化说明.md)。

规则级 Base Model 画像的生成阶段实验已推进到独立留出检查，结果见 [`docs/20260929_规则级模型画像与生成阶段正向验证实验.md`](docs/20260929_规则级模型画像与生成阶段正向验证实验.md)。当前 59/60 道独立探针绑定到 186 个“规则×能力”位置，规则级画像能实质改变选样；但在完全不同规则的留出检查中，aware 相对 agnostic 的分布误差改善中位数为 0，尚不支持未见规则推广。因此当前结论是“规则级方向成立，稀疏画像覆盖不足”，还不能据此声称完整的模型条件价值保真或训练效用。

面向非技术听众的会议材料包括 [`docs/20260930_会议一页稿_研究假设与形式化问题.md`](docs/20260930_会议一页稿_研究假设与形式化问题.md) 和作为答疑备份的 [`docs/20260930_小样预测全量价值_会议通俗说明.md`](docs/20260930_小样预测全量价值_会议通俗说明.md)，其中用桃子试吃和 CLL 医学规则案例解释研究问题、机理假设与最小形式化表达。

“模型相关因素是否必要”被拆为“模型是否需要进入价值计算”和“模型是否还需要进入小样生成”两个可独立检验的问题。A0—A4 对照、完整参考、评价指标、决策规则及 1.5B/7B 双模型可识别性实测见 [`docs/20261004_模型相关因素是否必要_实验验证方案.md`](docs/20261004_模型相关因素是否必要_实验验证方案.md)，筛查配置见 [`configs/valuation_cll_public_feasibility/conditional_value/model_relevance_necessity.screening.yaml`](configs/valuation_cll_public_feasibility/conditional_value/model_relevance_necessity.screening.yaml)。本地 Ollama 模型可通过 [`tools/run_ollama_model_probes.py`](tools/run_ollama_model_probes.py) 使用同一冻结探针协议生成第二模型画像。

面向非技术听众的简化说明见 [`docs/20261004_模型相关因素是否需要_问题与实验通俗说明.md`](docs/20261004_模型相关因素是否需要_问题与实验通俗说明.md)。该文档区分估值小样与知识填充集，并用现有双模型结果说明“已经得到支持的结论”和“仍需正式对照验证的结论”。

早期实验 1—15 的设计与当时状态见 [`docs/20261004_全部实验汇总_设计结果与当前状态.md`](docs/20261004_全部实验汇总_设计结果与当前状态.md)。其时间戳和实验协议不等于 rev1 论文的 Suite C；当前论文数值以 [`rev1 结果统一口径`](docs/rev1论文实验结果统一口径.md)为准。

当前论文结果的统一数值口径见 [`docs/rev1论文实验结果统一口径.md`](docs/rev1论文实验结果统一口径.md)，文件用途见 [`docs/结果文件索引.md`](docs/结果文件索引.md)。统一口径只摘录 rev1 论文报告的 Suite O 与 Suite C 数值，并分别标注实验单位、参照政策和指标分母；它不是逐题记录的独立复算结果。早期报告保留各自实验协议和产生时间，不能混入当前论文主结果。演示性数字不作为科研结果引用。

核心机理被提炼为“条件价值画像保真—学习实现—尺度传递机制”，见 [`docs/20260927_小样预测全量价值的机理提炼.md`](docs/20260927_小样预测全量价值的机理提炼.md)。该文档将 `数据 D × 模型 M × 任务 B` 展开为可学习性、规则级增量空间和任务相关性，并说明小样画像保真、学习实现、规模传递与最终排序预测之间的逐步因果关系。

先冻结并审查协议：

```powershell
ai-ready-value plan-ablations `
  --protocol configs/valuation_cll_public_feasibility/stage_ablation_protocol.yaml `
  --output-dir outputs/cll_stage_ablation_plan
```

代码执行顺序为：`plan-ablations` 冻结协议 → 为每个批次准备 `AblationBatchInput` → 用真实执行配置物化同生成器 cell → 按 `jobs.jsonl` 在 LLaMAFactory 服务器训练 → 从静态审计、逐题 base/tuned 预测和 proxy/FOM 批次效用计算 observations → `analyze-ablations` 做协议摘要、成对完整性和来源谱系聚类审计。执行配置从 [`stage_ablation_execution.template.yaml`](configs/valuation_cll_public_feasibility/stage_ablation_execution.template.yaml) 复制；其中 `<REPLACE_...>` 必须替换为预登记的真实模型状态、校准和摘要。代码要求 SHA-256 为 64 位十六进制，并重新核对本地 checkpoint／校准文件，模板本身不能直接运行。

同生成器的 100% cell 是机制参照，用于隔离覆盖选择、组织方式、规模、重复和噪声的作用；Fully Open Meditron 仍是独立的高预算外部参考，只回答小样排序能否预测生产全量排序。两类参考不会合并为同一个“全量组”。取得服务器逐阶段观测后，分析器分别输出 A1—A8 机制判定与 A9 FOM 跨流水线判定。当前配置是四个独立 lineage 的筛查协议，不承担稳定外推结论。完整代码流程和输入约束见 [`docs/消融实验代码执行说明.md`](docs/消融实验代码执行说明.md)；可执行技术协议见 [`docs/20260923_小样生成方法阶段级消融实验方案.md`](docs/20260923_小样生成方法阶段级消融实验方案.md)；面向汇报的实验设计与逐项依据见 [`docs/20260923_小样生成方法消融实验设计与设计依据.md`](docs/20260923_小样生成方法消融实验设计与设计依据.md)。

为 B01—B12 准备真实 batch input 后，完整物化 32 个同生成器 cell：

```powershell
$batchArgs = Get-ChildItem inputs/stage_ablation/*.yaml |
  Sort-Object Name |
  ForEach-Object { "--batch-input"; $_.FullName }

python tools/value_data.py materialize-ablations `
  --protocol configs/valuation_cll_public_feasibility/stage_ablation_protocol.yaml `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml `
  --execution configs/valuation_cll_public_feasibility/stage_ablation_execution.yaml `
  @batchArgs `
  --output-dir outputs/cll_stage_ablation_execution
```

`AblationBatchInput` 中的候选知识、覆盖义务、能力映射、三种配对监督形式和 A6 规则干预必须来自真实批次。A1 人工金标、模型 checkpoint 与独立校准、诊断探针以及 FOM 全量结果也都是外部实验工件，仓库不会用模板值代替。

在服务器生成 LLaMAFactory bundle：

```bash
python tools/value_data.py prepare-ablation-llamafactory \
  --jobs outputs/cll_stage_ablation_execution/jobs.jsonl \
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml \
  --profile configs/valuation_cll_public_feasibility/llamafactory/server_profile.yaml \
  --llamafactory-root "$LF_ROOT" \
  --output-dir outputs/cll_stage_ablation_llamafactory
```

训练和评测返回后，使用 `collect-ablation-observations` 从静态选择审计、逐题预测或 proxy/FOM 批次效用计算已登记指标。该命令要求 `--kind selection|prediction|ranking`、对应的 `--study` 和 `--cell`，可重复传入 `--metric`，并以 `--execution-manifest outputs/cll_stage_ablation_execution/execution_manifest.json` 绑定整次运行。汇总 observation JSONL 后执行：

```powershell
python tools/value_data.py analyze-ablations `
  --protocol configs/valuation_cll_public_feasibility/stage_ablation_protocol.yaml `
  --observations outputs/server/ablation_observations.jsonl `
  --output-dir outputs/cll_stage_ablation_report
```

`select-ablation-fraction` 为每个冻结候选池派生 25%/50%/100% 的批次相对预算，默认拒绝全选并记录候选池与选择比例；100% 同生成器参考需要显式使用 `--allow-full-candidate-pool`。该入口只服务消融，原正式生产选择配置和哈希语义保持不变。

## LLaMAFactory 服务器训练入口

训练适配器把 12 个批次 × 7 个实验臂整理为 84 份不可变 SFT 数据包，再按 3 个训练种子生成 252 个独立 LoRA 作业。三个种子共享同一份转换后 SFT，只改变 `seed` 与 `data_seed`。每份数据包必须包含独立的 `generation_manifest.json`：代理臂登记覆盖生成配方、生成器实现 SHA-256 和选择清单，全量臂登记 Fully Open Meditron 配方、上游运行清单和完整遍历报告。训练准备阶段会拒绝两条支路共用同一 SFT 文件、生成器身份互换或原始批次哈希不一致。

先生成覆盖小样：

```powershell
python tools/value_data.py generate-proxy-sft `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml `
  --blueprint configs/valuation_cll_public_feasibility/capability_blueprint.yaml `
  --selection-config configs/valuation_cll_public_feasibility/proxy_selection/coverage_quality.yaml `
  --output-dir outputs/cll_sft/coverage_quality
```

Fully Open Meditron 在外部完成整批生成后，逐批导入规范化结果：

```powershell
python tools/value_data.py import-fully-open-meditron `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml `
  --batch-id B01 `
  --batch-content-sha256 <batch_suite.frozen.yaml中的B01哈希> `
  --eligible-units outputs/fom/B01/eligible_units.jsonl `
  --fully-open-meditron-output outputs/fom/B01/normalized_fom.jsonl `
  --fully-open-meditron-run-manifest outputs/fom/B01/fully_open_meditron_run_manifest.json `
  --output-dir outputs/cll_sft/operational_full/B01
```

规范化的 Fully Open Meditron JSONL 每行需要保留 `semantic_unit_id`、`instruction`、`response`、`evidence_spans`、`source_ids` 和 `task_type`。导入器不会调用覆盖选择器；没有生成合格样本的知识单元会被记录为失败，而不会从完整遍历中消失。正式运行要先在 FOM 作业端写出 `fully_open_meditron_run_manifest.json`（`fully-open-meditron-run-manifest-v1`），再导入数据；导入器会核对其工具 adapter/repository/config 哈希、生成/教师配方、批次与合同身份、eligible 输入哈希、规范化输出哈希以及“未访问 Benchmark/未使用代理选择器/完整遍历”声明。

当前仓库合同中的 FOM revision 和 reference teacher revision 仍带 `example-*` 占位符。服务器正式运行前必须替换成实际 checkout／模型 revision 并重新执行 `plan-experiment`；导入器会主动拒绝占位 revision，避免把普通 JSONL 误登记为正式 FOM 产物。

`scaffold-llamafactory --sft-root outputs/cll_sft` 使用 `<SFT_ROOT>/<arm_label>/<batch_id>/` 布局。六个代理臂分别以对应的选择配置运行一次 `generate-proxy-sft`；上例会直接生成 `coverage_quality/B01`—`coverage_quality/B12`。全量参考则逐批导入 `operational_full/B01`—`operational_full/B12`。

服务器上的主要命令为：

```bash
python tools/value_data.py scaffold-llamafactory \
  --manifest "$PLAN_DIR/experiment_manifest.json" \
  --sft-root "$SFT_ROOT" \
  --output "$CONTROL_DIR/sft_registry.yaml"

# 在一张与正式训练同型号的 GPU allocation 内执行；Slurm 会设置该变量。
CUDA_VISIBLE_DEVICES=0 python tools/value_data.py prepare-llamafactory \
  --manifest "$PLAN_DIR/experiment_manifest.json" \
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml \
  --registry "$CONTROL_DIR/sft_registry.yaml" \
  --profile configs/valuation_cll_public_feasibility/llamafactory/server_profile.yaml \
  --llamafactory-root "$LF_ROOT" \
  --output-dir "$BUNDLE_DIR"

CUDA_VISIBLE_DEVICES=0 python tools/value_data.py run-llamafactory \
  --jobs "$BUNDLE_DIR/llamafactory_jobs.json" \
  --job-index 0 \
  --resume
```

`prepare-llamafactory` 必须在最终 Linux 训练服务器的单 GPU compute allocation 内执行，因为 bundle 会冻结绝对路径、Python 解释器、LLaMAFactory checkout、项目关键代码哈希、依赖版本、CUDA／GPU／driver 指纹和模型 tokenizer。Slurm 环境应保留调度器设置的 `CUDA_VISIBLE_DEVICES`；上面的 `0` 只表示裸机单卡示例。训练只产生 checkpoint 与待评测请求，完整实验还需要 OpenCompass 评测和 `ExperimentTrialScore`。项目组已完成本轮服务器效用验证；逐 trial 结果与哈希仍需同步回本地仓库。

`scaffold-llamafactory` 只生成 84 个登记位置。每个代理包需提供 accepted SFT、selection manifest、proxy units、coverage obligations 和 dispositions；每个参考包需提供 accepted SFT、eligible units、dispositions 和 full-traversal report。准备阶段会现场重放选择与完整遍历审计。

## 环境与安装

本机已创建 Conda 环境 `aiReady`，Python 3.11.16，解释器为 `D:/Anaconda3/envs/aiReady/python.exe`。后续运行和安装都使用该环境。

```powershell
conda activate aiReady
cd D:\pythonCode\research\AI-ready\ai-ready-benchmark
python tools/process_data.py doctor
```

在新机器重建环境：

```powershell
.\tools\setup_env.ps1 -Dev
```

脚本只创建缺失的 `aiReady`，检查 Python 3.11，并把依赖安装到该环境。直接依赖见 `pyproject.toml`，Windows/Python 3.11 的精确环境快照见 `requirements-win-py311.lock`。Data-Juicer 的包名是 `py-data-juicer`，导入名是 `data_juicer`；这里只安装所需依赖，不安装包含平台专属组件的全部 extras。

## 数据估值主线

本节说明通用数据估值接口。若只复现上面的 CLL/SLL 实验，应直接使用 `configs/valuation_cll_public_feasibility/` 中的实际冻结配置。`examples/valuation_cll/` 是用于建立新实验的模板，其中 `pending://` 来源、内容哈希、外部工具 revision 和 argv 必须替换并冻结。

```powershell
python tools/value_data.py validate `
  --batches examples/valuation_cll/batch_suite.yaml `
  --contract examples/valuation_cll/production_contract.yaml

python tools/value_data.py freeze-batches `
  --batches configs/valuation/cll_batches.yaml `
  --output outputs/cll_batches.frozen.yaml

python tools/value_data.py plan-reference `
  --batches outputs/cll_batches.frozen.yaml `
  --contract examples/valuation_cll/production_contract.yaml `
  --output-dir outputs/cll_reference_plan
```

示例 `batch_suite.yaml` 中使用 `pending://`，所以第一条命令用于审查模板；正式运行时先把它复制为 `configs/valuation/cll_batches.yaml`，替换成真实本地数据和许可登记，再执行 `freeze-batches`。冻结命令会复算批次内容、来源版本和构造配方的哈希；已有哈希与本地内容不一致时直接失败。

`plan-reference` 要求所有批次已经冻结，并再次核对本地内容。它只生成可审查的全批次、全种子执行计划，不会伪装成已运行外部训练。Production Contract 中配置实际 argv 后，仍应由受控作业系统执行 Data-Juicer、Fully Open Meditron、OpenDataArena／LLaMA-Factory 和 OpenCompass／EvalScope，并按标准 JSONL 接口回传结果。

从 L1 的来源证据挖掘语义单元：

```powershell
python tools/value_data.py mine `
  --records outputs/my_batch/analyzed/records.jsonl `
  --activities examples/valuation_cll/activities.yaml `
  --batch-id B01 `
  --output-dir outputs/B01_semantic
```

`mine` 只标记来源文本中实际出现的条件、例外、时效和不确定性片段，不生成题目，也不构造机械的主题×任务×难度笛卡尔积。准备带批次内容哈希、embedding、token、质量分层和 `group_id` 的 `ProxyUnit` JSONL 后运行：

```powershell
python tools/value_data.py select `
  --units data/B01_proxy_units.jsonl `
  --obligations outputs/B01_semantic/coverage_obligations.jsonl `
  --config examples/valuation_cll/proxy_selection.yaml `
  --output outputs/B01_proxy_selection.json
```

汇入代理与高预算试验的逐批、逐种子得分后计算估值指标：

```powershell
python tools/value_data.py metrics `
  --batches outputs/cll_batches.frozen.yaml `
  --contract examples/valuation_cll/production_contract.yaml `
  --trials results/trials.jsonl `
  --proxy-method coverage_quality --proxy-budget low `
  --reference-method fully_open_meditron --reference-budget high `
  --top-k 1 3 `
  --output-dir outputs/valuation_metrics
```

详细接口和科学约束见 [`docs/数据估值实现说明.md`](docs/数据估值实现说明.md)。

## L1 与旧 L2 兼容支线

```powershell
python tools/process_data.py run --config configs/l1.yaml --output-dir outputs/my_run
python tools/process_data.py search --config configs/l1.yaml --output-dir outputs/my_run --query "提前还贷需要提前多少天申请？" --top-k 3
```

输出目录必须不存在或为空，程序不会覆盖已有实验。分阶段运行：

```powershell
python tools/process_data.py parse   --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py clean   --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py analyze --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py index   --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py cluster --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py demand  --config configs/l1.yaml --output-dir outputs/staged_run
python tools/process_data.py publish --config configs/l1.yaml --output-dir outputs/staged_run
```

每阶段读取前序落盘产物。失败会保留状态和已产生文件；重跑完整流程应换新目录。

## 可讲解案例

`examples/leukemia_beginner_case/` 是一套面向非技术听众的白血病教学案例，包含原始文档、配置、实际 L1→L2 运行结果、五分钟讲解稿和人工编写的未来 L3 题目示意。医学部分来自公开权威资料的概括，版本管理制度完全虚构，不用于诊疗。

## 输入、登记与版本

递归处理 PDF、DOCX、HTML、HTM、Markdown、UTF-8 TXT、PNG、JPEG 和 TIFF。旧式 DOC 需先转成 DOCX。将文档放入独立只读输入目录，在配置中设置 `input_dir`，并用 `registry.metadata_file` 指向文档登记表。

```yaml
documents:
  - path: 贷款规则_示例.md
    logical_key: policy/loan-rule
    title: 贷款规则
    version: "2026.1"
    effective_from: "2026-01-01"
    status: active
    permissions: [research, loan-team]
    business_weight: 1.5
    risk_weight: 2.0
```

三个 ID 分工：

- `document_id`：由 `logical_key` 派生，同一逻辑文档跨版本稳定。
- `document_version_id`：由逻辑文档、版本和文件 SHA-256 派生，版本内容变化就变化。
- `DocumentUnit.id`：由文档版本、原始结构单元和文本派生，可追溯到该版本中的原始证据。

文件 SHA-256 用于精确重复；文档 SimHash 用于近版本关系；单元层调用 Data-Juicer 做精确去重和 MinHash 近重复标记。同内容的不同业务版本不会被跨版本删除，而会保留 `previous_version_id`、`exact_duplicate_of` 或 `near_duplicate_version_ids`。

## L1：解析、证据与存储

`parsing.backend: auto` 的路由是：文本 PDF 优先 PyMuPDF，DOCX 使用 python-docx，HTML/Markdown/图片和扫描 PDF 使用 Docling；检测到表格或复杂多栏的 PDF 使用 `complex_pdf_backend`，默认 Marker。Marker 的结构结果同时保存在 `parsed/attachments/`，再统一转换为 DoclingDocument 进入同一分块接口。扫描件保持 Docling + RapidOCR，保存可获得的页码与 bounding box。

Marker 2.0 的代码许可证和模型权重许可证不同，企业使用前应核对其[官方许可与商业条款](https://github.com/datalab-to/marker#commercial-usage)。

结构单元保留标题上下文、条款/正文/表格/列表/脚注标签、交叉引用、Docling `item_ref`、页码和坐标。可用 `sliding_window_units` 建检索窗口，但窗口只记录 `gold_source_unit_ids`，gold evidence 始终回指原始结构单元；窗口副本不参与主题发现。

L1 的正式接口：

| 产物 | 用途 |
| --- | --- |
| `documents.parquet` | 逻辑文档、版本、URI、权限、生效期、重复/版本关系 |
| `document_units.parquet` | 原始单元与窗口、副本关系、结构标签、来源定位、六维文档难度 |
| `parsed/documents/*.json` | 完整 DoclingDocument |
| `parsed/attachments/*.marker.json` | Marker 解析附件与配置（仅走 Marker 时） |
| `index/` | 与 embedding 模型名、revision、维度和语料哈希绑定的 FAISS 索引 |
| `catalog.duckdb` | 对根目录 Parquet 表的只读视图和数据集信息 |
| `snapshot.json` | 数据集版本、处理配置哈希及产物内容哈希 |
| `mlruns/mlflow.db` | 本地 MLflow 实验、阶段指标与 manifest |

Parquet + DuckDB 用于结构表，FAISS 用于向量，MLflow 使用本地 SQLite 后端；`snapshot.json` 是当前单机原型中的 DVC 等价内容快照。生产环境可把快照纳入真正的 DVC/对象存储流程。

## 文档难度与题目难度

二者在代码和表中分开：

- `DifficultyProfile` 从结构、语言、知识密度、依赖、时间、歧义/冲突六类信号得到 `document_difficulty`。初始权重只用于路由和分层，不是论文结论。
- `TaskRoute` 单独记录 evidence hops、文档数、推理步数、隐含性、干扰项、时间推理、输出约束、多轮轮数，并生成 `predicted_score/level`。
- `DemandCell.target_level` 是希望生成的难度；`TaskRoute.predicted_level` 是生成前代理预测；`empirical_score` 预留给后续 anchor systems + IRT 校准。

证据束不是简单取单段难度：`evidence_bundle_difficulty` 汇总各单元六维特征，并显式加入多单元、跨文档依赖。简单文档仍可通过多证据、干扰项、版本或输出约束形成中/难任务。

## L2：从文档结构到三元需求点

主题、任务、难度不是一次 LLM 分类得到，而是分层构造：

1. HDBSCAN/BERTopic 给出主题候选，LLM 只预留用于命名和合并建议。`cluster_id=-1` 会转为独立长尾主题，不丢弃。
2. 业务/风险关键词规则可创建低频人工主题，即使它不足以形成聚类。
3. 每个主题内构造单证据和代表性多证据 `KnowledgeAnchor`，记录文档、能力及证据束难度。
4. 六类 `task_type` 固定为 `factual_qa`、`multi_turn`、`summarization`、`judgment`、`comparison`、`extraction`；`capability` 单独描述条款理解、条件推理、时序判断、异常处理、跨文档综合等。
5. 规则先检查证据是否满足每类任务的最小条件，并输出全部 `TaskRoute`（含拒绝理由）；只对可行路线形成 `(topic, task_type, target_level)`。

因此落盘的是有效集合：

```text
Z_valid = {(topic, task, difficulty) | feasibility(...) = true}
```

不会给每个主题机械生成 `6 × 3` 题目。完整实现说明见 `docs/L2需求建模说明.md`。

## 缺口、优先级与覆盖选点

`DemandCell` 记录当前有效题数量，以及未覆盖、文档覆盖需求、长尾度、业务重要度、风险、文档难度—任务路由适配度、目标任务配比、目标难度配比、冗余等信号。低难证据优先事实/抽取/局部摘要，中等证据提高判断/对比/多轮的路由权重，高难证据优先条件、跨文档和版本类任务，但路由权重不会被误当成硬性可行性。将已有 benchmark 配置为 Parquet 或 JSONL 后，会按 `topic_id + task_type + difficulty` 统计有效题并重新计算缺口：

```yaml
l2:
  existing_benchmark_file: ../data/current_benchmark.parquet
  target_questions_per_cell: 2
```

行至少包含 `topic_id`、`task_type`、`difficulty`（或 `target_level`）；建议同时提供 `document_id` / `document_ids` 或 `evidence[].document_id`，用于计算文档覆盖缺口。`valid: false` 的题不计入覆盖。所有权重和目标配比写入 manifest，便于 L5 检验“覆盖信号是否忠实于下游评测效用”。

选点器支持 `facility_location`、`random`、`size_proportional`、`cluster_centroid`。facility-location greedy 同时考虑语义代表性、DemandCell 优先级和新格子增益；后三者作为实验基线。它们是可替换的经典选择器，不被声明为方法创新。

## 目录与主要产物

```text
ai-ready-benchmark/
├── ai_ready/
│   ├── core/          # schema、登记、流水线
│   ├── format/        # 解析、结构恢复和分块
│   ├── ops/           # Data-Juicer 清洗与去重
│   ├── difficulty/    # 文档/证据束/任务难度代理
│   ├── embedding/     # BGE / Qwen
│   ├── retrieval/     # FAISS
│   ├── clustering/    # HDBSCAN / BERTopic
│   ├── l2/            # 旧 benchmark 构造支线，保留兼容
│   ├── valuation/     # 批次、合同、语义挖掘、代理选点、质量门和效用指标
│   └── storage/       # Parquet、DuckDB、快照、MLflow
├── configs/
├── demos/documents/   # 虚构示例
├── tools/
├── tests/
└── docs/
```

一次完整运行还会生成 `topics.parquet`、`knowledge_anchors.parquet`、`task_catalog.parquet`、`task_routes.parquet`、`demand_cells.parquet`、`selected_anchors.parquet` 和 `knowledge_edges.parquet`。最后一张表把文档版本包含单元、主题包含成员、anchor 使用证据、任务支持/拒绝、cell 候选等关系显式化；`demand/` 中有相同内容的 JSONL 便于人工调试。

## 开发与验证

```powershell
conda activate aiReady
python -m pytest -q
python -m pytest -q --run-integration
python -m black --check ai_ready tools tests
python -m isort --check-only ai_ready tools tests
python -m flake8 ai_ready tools tests
python -m pip check
```

默认测试不下载大模型；单元测试覆盖现有 L1/L2 兼容支线和数据估值核心约束，集成测试覆盖真实 BGE、Docling OCR、Marker 2.0 和完整旧 L1→L2 流程。详见 `docs/运行验证记录.md`。

## 当前研究边界

- 当前是单机研究原型，不提供百万级分布式、细粒度权限执行、在线增量聚类或生产对象存储。
- 当前仓库不内嵌 LLaMAFactory 或 OpenCompass；训练适配器生成它们可消费的工件，核对版本、模板、token、运行时和输出哈希，实际模型计算仍由服务器上的 LLaMAFactory 执行。
- `examples/valuation_cll/` 中的 Production Contract 仍是建立新实验的模板。公开 CLL 可行性配置已绑定 LLaMAFactory 与基础模型 revision；生成器和 OpenCompass 的真实服务器运行仍需在正式执行前冻结。
- 静态覆盖和质量分布是机制证据；只有代理试训对高预算训练的跨批次排序、Top-k 与后悔值结果才能支持估值主张。
- 8—12 个批次只用于可行性和方法筛选，不用于声称稳定相关性；同源受控批次不能当作独立供应商。
- 旧 L2 的 DemandCell 和 difficulty 逻辑保留用于已有 benchmark 构造实验，不参与新估值主线的高预算参考。

## 官方参考

- [Data-Juicer 文档](https://datajuicer.github.io/data-juicer/zh_CN/main/docs_index_ZH.html) · [GitHub](https://github.com/datajuicer/data-juicer)
- [Fully Open Meditron](https://github.com/EPFLiGHT/FullyOpenMeditron) · [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) · [Bonito](https://github.com/BatsResearch/bonito)
- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) · [EvalScope](https://github.com/modelscope/evalscope) · [OpenCompass](https://github.com/open-compass/opencompass)
- [DoclingDocument 结构](https://docling-project.github.io/docling/concepts/docling_document/#basic-structure) · [Hybrid chunking](https://docling-project.github.io/docling/_generated/examples/hybrid_chunking/) · [整页 OCR](https://docling-project.github.io/docling/_generated/examples/full_page_ocr/)
- [Marker](https://github.com/datalab-to/marker)
- [DuckDB 读取 Parquet](https://duckdb.org/docs/stable/data/parquet/overview) · [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) · [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [BGE small zh v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) · [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
