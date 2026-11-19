# Shein-F4

F4跨境物流时效预测代码仓库。下列结构列出纳入Git版本管理的文件及其作用。

## 文件结构与作用

```text
Shein-F4/
├── README.md                         # 仓库文件结构与职责说明
├── AGENTS.md                         # 项目协作、数据使用和代码组织规则
├── .gitignore                        # Git忽略规则
├── pyproject.toml                    # Python包元信息、依赖与构建配置
├── requirements.txt                  # 项目直接依赖清单
├── requirements.lock.txt             # 固定版本的环境依赖清单
│
├── configs/                          # 数据、切分及实验参数
│   ├── data.yaml                     # 原始数据路径、表结构和分块处理参数
│   ├── splits.yaml                   # 时间窗口、标签截止与独立留后切分配置
│   ├── splits_deploy_C.yaml          # C时点构建模型的独立切分及原冻结切分引用
│   ├── experiments/                  # 准备阶段预检配置
│   │   ├── baseline.yaml             # B1：承运商常数基线配置
│   │   ├── quantile.yaml             # M1：LightGBM Q90/Q97分位数模型配置
│   │   └── calibrated.yaml           # M2：分位数模型与承运商残差校准配置
│   └── training/                     # 本地训练配置与批次计划
│       ├── global_baseline.yaml      # B0：全局常数基线训练配置
│       ├── baseline.yaml             # B1：承运商常数基线训练配置
│       ├── quantile.yaml             # M1：双分位数模型训练配置
│       ├── calibrated.yaml           # M2：双分位数与承运商校准训练配置
│       ├── calibrated_baseline.yaml  # B1C：承运商常数与同口径残差校准配置
│       ├── first_pass_20261118.yaml   # 首轮双开发窗口的实验队列及资源限制
│       ├── fair_calibration_20261118.yaml # 双开发窗口的校准常数公平对照队列
│       ├── deploy_C_baseline.yaml    # C协议的B1C常数与校准对照配置
│       ├── deploy_C_quantile.yaml    # C协议的M1双分位数模型配置
│       ├── deploy_C_calibrated.yaml  # C协议的M2双分位数与校准配置
│       └── deploy_C_20261119.yaml     # C协议双窗口六组实验及资源限制
│
├── src/
│   └── f4/                           # 核心Python包
│       ├── __init__.py               # Python包标识与说明
│       ├── common.py                 # 项目路径、文件摘要、执行快照与原子写入
│       ├── data.py                   # 分块读取、核心表转换与数据审计
│       ├── features.py               # 字段白名单、数值及时间特征、类别编码
│       ├── splits.py                 # 固定时间切分、标签时点与留后隔离
│       ├── models.py                 # B0/B1基线与LightGBM分位数模型
│       ├── calibration.py            # 按承运商独立进行残差校准
│       ├── metrics.py                # 评分计算、包裹键对齐及分组诊断
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
│   ├── audit_data.py                 # 数据结构与开发标签审计入口
│   ├── diagnose_development.py       # 开发队列标签成熟度与时间特征偏差诊断
│   ├── make_splits.py                # 生成并冻结时间切分键
│   ├── backtest.py                   # 实验预检与受控回测入口
│   ├── train_local.py                # 串行训练、执行快照、内存监控及状态记录
│   ├── evaluate.py                   # 对齐预测与标签并计算评分
│   ├── validate_submission.py        # 检查包裹键、匹配关系及预测值格式
│   ├── smoke_test.py                 # 运行合成冒烟测试并汇总结果
│   └── summarize_preparation.py      # 核验准备材料与测试快照并汇总状态
│
├── tests/                            # 合成数据测试
│   ├── test_data.py                  # 数据转换、键完整性及审计边界
│   ├── test_download_bootstrap.py    # 下载目录初始化、清单保留及文件锁
│   ├── test_development_diagnostics.py # 成熟度分母、留后读取边界及预测匹配
│   ├── test_splits.py                # 时间切分、标签成熟度及留后隔离
│   ├── test_models.py                # 特征编码、模型、校准及保存加载
│   ├── test_metrics.py               # 官方评分一致性、边界与键对齐
│   ├── test_backtest.py              # 端到端回测、执行保护及产物管理
│   ├── test_train_local.py           # 训练队列、源码漂移及资源故障检查
│   └── test_cli.py                   # 命令行评分与提交文件校验
│
└── official/
    └── tools/
        └── score.py                  # 官方评分器原样副本，用于评分一致性核对
```

## 模块关系

`configs/`定义参数，`scripts/`提供运行入口，`src/f4/`实现核心逻辑，`tests/`验证模块行为与端到端流程。`official/tools/score.py`提供官方评分参考，`src/f4/metrics.py`负责项目内的评分封装与诊断。
