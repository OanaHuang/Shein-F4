# Shein-F4

F4跨境物流时效预测代码仓库。下列结构列出纳入Git版本管理的文件及其作用。

模型选型、各模型在后续流程中的作用及阶段产出见[《模型选择与后续流程说明》](docs/model_workflow.md)。方案文档不代表实验已完成；实际状态以带时间和来源的本机JSON报告为准。

## 结果与诊断

代码职责、首轮分数、局限和本次改动见[《代码、首轮结果与改进记录》](docs/code_and_results_review.md)。已完成[第4轮dev_2复核](docs/dev2_comparison_results.md)、[第5轮CatBoost对照](docs/catboost_comparison_results.md)及[第6轮成熟历史下限对照](docs/mature_history_floor_results.md)。最新LightGBM＋校准＋历史下限的A为167.34／151.27，低于原M2的421.53／341.90及同样加下限的承运商方案171.84／155.11。代价是平均承诺增加66／85小时、逐包MAE增大，约96%预测被历史下限抬高。该方案是当前开发A最优候选，尚未冻结最终方案；两窗口存在重叠，最终留后未使用。

只读复核已有运行（不会重新拟合，报告ID不可重复）：

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_pilot.py --pilot-report reports/local_pilot/local_dev1_20261118_01/summary.json --report-id review_002
```

诊断会验证源数据、固定键、运行清单、预测及校准器摘要，输出到`reports/diagnostics/<report-id>/diagnostics.json`。后续回测自动保存阶段标签分布、总体与分组覆盖、校准前后诊断。`configs/experiments/baseline_calibrated.yaml`保留准备模板；用户后续授权的副本`configs/local_pilot/baseline_calibrated.yaml`已完成dev_1真实对照，A为356.72，优于已有M2的421.53。仅代表这个开发窗口，详见复盘第6节。

## 本地首轮开发实验

用户于2026-11-18授权先在本机运行。本轮范围为固定 `dev_1` 上的B1承运商基线、M1 LightGBM和M2 LightGBM＋校准，最多4线程、串行执行，不搜索参数、不评估最终留后。
独立配置位于 `configs/local_pilot/`，其他准备配置保持锁定。执行入口：

```powershell
.\.venv\Scripts\python.exe scripts/run_local_pilot.py --run-id local_dev1_001 --execute-training
```

每次使用新的ID。入口记录端到端耗时，实际Python工作进程在回测结束时自行记录Windows工作集峰值和提交内存峰值（不代表整机内存，避免误测虚拟环境启动器）。汇总位于 `reports/local_pilot/<run-id>/summary.json`，模型、预测和评分分别保存在相应的 `artifacts/runs/`、`outputs/runs/`、`reports/runs/`。开发评分不是比赛测试集成绩；`reports/preparation_status.json`仅记录此前准备阶段的状态。

## Windows 准备流程

支持 Python 3.10 及以上。新机器先建立项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "catboost>=1.2,<2"
```

完整准备入口（不拟合真实数据、不生成正式分数）：

```powershell
.\.venv\Scripts\python.exe scripts/run_preparation.py --archive D:\Desktop\F4.zip --run-id prep_001
```

该命令按顺序检查环境、运行独立合成测试、校验并解压ZIP、生成核心缓存、审计共有字段、冻结留后与开发切分、审计开发标签，最后对全部实验配置执行dry-run。已有核心缓存仅在源文件和缓存摘要一致时复用；每次使用新的准备ID。失败原因与各步骤日志保存在`reports/preparation/<run-id>/`，成功状态写入`reports/preparation_status.json`。

若只需要合成检查：

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

`configs/experiments/catboost.yaml`提供CatBoost＋校准，`remaining.yaml`提供LightGBM剩余时长＋校准；两个模板仍为准备模式。CatBoost的授权副本`configs/local_catboost/catboost.yaml`已完成第5轮dev_1、dev_2固定参数对照。剩余时长遇到时间倒挂、缺失或非正剩余标签会失败，不删除样本。历史统计特征和TabM尚未接入。

`requirements.lock.txt`保留原有环境快照；本次Windows环境的精确版本单独记录于`requirements.windows-py310.lock.txt`。环境检查不会覆盖任一锁定文件。原始数据、缓存和本机报告按`.gitignore`留在本机。

## 文件结构与作用

```text
Shein-F4/
├── README.md                         # 仓库文件结构与职责说明
├── AGENTS.md                         # 项目协作、数据使用和代码组织规则
├── .gitignore                        # Git忽略规则
├── pyproject.toml                    # Python包元信息、依赖与构建配置
├── requirements.txt                  # 项目直接依赖清单
├── requirements.lock.txt             # 固定版本的环境依赖清单
├── requirements.windows-py310.lock.txt # 本次Windows/Python 3.10环境快照
├── docs/
│   ├── model_workflow.md              # 模型职责、接入顺序、验证标准与后续流程
│   └── code_and_results_review.md     # 代码职责、首轮结果、诊断与改进记录
│
├── configs/                          # 数据、切分及实验参数
│   ├── data.yaml                     # 原始数据路径、表结构和分块处理参数
│   ├── splits.yaml                   # 时间窗口、标签截止与独立留后切分配置
│   └── experiments/
│       ├── baseline.yaml             # B1：承运商常数基线配置
│       ├── baseline_calibrated.yaml  # B1＋相同独立校准的待测对照
│       ├── quantile.yaml             # M1：LightGBM Q90/Q97分位数模型配置
│       ├── calibrated.yaml           # M2：分位数模型与承运商残差校准配置
│       ├── catboost.yaml             # M2：CatBoost双分位数与独立校准
│       └── remaining.yaml            # M2：预测剩余时长后还原总时长
│
├── src/
│   └── f4/                           # 核心Python包
│       ├── __init__.py               # Python包标识与说明
│       ├── common.py                 # 项目路径、文件摘要和原子写入工具
│       ├── data.py                   # 分块读取、核心表转换与数据审计
│       ├── features.py               # 字段白名单、数值及时间特征、类别编码
│       ├── splits.py                 # 固定时间切分、标签时点与留后隔离
│       ├── models.py                 # B0/B1基线与LightGBM、CatBoost分位数模型
│       ├── targets.py                # 总时长/剩余时长转换与异常检查
│       ├── platform_utils.py         # 跨平台文件锁、分块哈希和UTC时间
│       ├── calibration.py            # 按承运商独立进行残差校准
│       ├── metrics.py                # 评分计算、包裹键对齐及分组诊断
│       ├── diagnostics.py            # 阶段分布、偏差、总体与分组覆盖
│       ├── backtest.py               # 串联特征、模型、校准、评分与实验产物
│       └── safety.py                 # 执行阶段与输入范围检查
│
├── scripts/                          # 命令行入口和辅助脚本
│   ├── download_official.sh          # 官方数据下载器的Shell入口
│   ├── fetch_official.py             # 初始化下载状态、断点续传及调用校验解压
│   ├── verify_archive.py             # 压缩包大小、CRC、摘要和解压路径校验
│   ├── install_environment.py        # 后台安装依赖、复用缓存及失败重试
│   ├── monitor_environment.py        # 监控已有依赖安装进程并核验环境
│   ├── check_environment.py          # 依赖完整性与合成小样例环境检查
│   ├── prepare_data.py               # 核心数据表转换入口
│   ├── run_preparation.py            # 本地ZIP到缓存、冻结切分和dry-run的串行入口
│   ├── audit_data.py                 # 数据结构与开发标签审计入口
│   ├── make_splits.py                # 生成并冻结时间切分键
│   ├── backtest.py                   # 实验预检与受控回测入口
│   ├── run_local_pilot.py            # 已授权开发窗口、方案串行运行与资源记录
│   ├── diagnose_pilot.py             # 核验已有产物并只读诊断，不重新拟合
│   ├── evaluate.py                   # 对齐预测与标签并计算评分
│   ├── validate_submission.py        # 检查包裹键、匹配关系及预测值格式
│   ├── smoke_test.py                 # 运行合成冒烟测试并汇总结果
│   └── summarize_preparation.py      # 核验准备材料与测试快照并汇总状态
│
├── tests/                            # 合成数据测试
│   ├── test_data.py                  # 数据转换、键完整性及审计边界
│   ├── test_download_bootstrap.py    # 下载目录初始化、清单保留及文件锁
│   ├── test_splits.py                # 时间切分、标签成熟度及留后隔离
│   ├── test_models.py                # 特征编码、模型、校准及保存加载
│   ├── test_metrics.py               # 官方评分一致性、边界与键对齐
│   ├── test_diagnostics.py           # 总体覆盖与小组短板、ME/MAE区别
│   ├── test_backtest.py              # 端到端回测、执行保护及产物管理
│   ├── test_cli.py                   # 命令行评分与提交文件校验
│   └── test_workflow.py              # 新模型、目标转换和Windows文件锁检查
│
└── official/
    └── tools/
        └── score.py                  # 官方评分器原样副本，用于评分一致性核对
```

## 模块关系

`configs/`定义参数，`scripts/`提供运行入口，`src/f4/`实现核心逻辑，`tests/`验证模块行为与端到端流程。`official/tools/score.py`提供官方评分参考，`src/f4/metrics.py`负责项目内的评分封装与诊断。


## 已授权的dev_2对照

`configs/local_dev2/`仅授权固定dev_2的B1＋独立校准、原M2和收缩M2，参数与dev_1一致。实际运行批次为`local_dev2_20261118_01`，不要重用该ID。运行入口已支持`--fold`、`--config-dir`和`--experiments`；复用模型配置中的`reuse_model_run`必须指向同一窗口、相同来源与参数的原M2。最终留后仍锁定。
