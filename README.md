# F4 跨境物流时效预测

本项目用于实施F4跨境物流时效预测任务。2026年11月17日已完成环境、全量核心缓存、结构审计、固定切分及开发标签审计；代码与合成冒烟测试已经准备。**尚未对真实F4数据拟合B0/B1或LightGBM，未报正式验证分数。** 当前阶段约束见`AGENTS.md`，本机生成的可复核状态见`reports/preparation_status.json`。

## Git协作

以本目录为Git仓库根，远端为 `https://github.com/OanaHuang/Shein-F4.git`。代码、配置、测试、依赖清单和官方评分器副本纳入版本管理；数据集、虚拟环境、缓存、切分键、模型、预测、生成报告、官网模板下载及本地 `.env` 配置由 `.gitignore` 排除。

首次推送前先确认GitHub登录状态和远端分支。远端已有提交时先获取并整合历史，不使用强制推送。日常修改在项目目录运行 `git status` 检查文件，再提交和推送；Git只同步纳入版本管理的文件。

全新克隆不包含本机数据或准备报告，需要安装依赖、重新下载数据并执行下面的准备流程。下载脚本会初始化所需目录和下载清单，不会沿用其他机器的下载完成状态。仓库同步不会解除正式训练限制。

## 目录

| 路径 | 用途 |
|---|---|
| `data/raw/` | 官方ZIP与原始解压文件，核验后保持不变 |
| `data/processed/` | 核心Parquet与按需构建的历史缓存 |
| `official/tools/` | 官方评分器原样副本 |
| `docs/templates/` | 官网作品说明单及AI协作记录模板 |
| `configs/` | 数据、固定切分及B1/M1/M2实验配置 |
| `scripts/`、`src/f4/` | CLI入口，以及数据、特征、模型、校准、评分与回测逻辑 |
| `tests/` | 合成数据的评分、键、泄漏边界、模型与端到端冒烟 |
| `artifacts/` | 切分键、编码器、模型及校准参数 |
| `reports/` | 数据清单、审计、实验和下载记录 |
| `outputs/` | 验证预测及最终CSV |

## 当前数据与验证

- 原始训练集5,267,074行、67列；测试集911,415行、23列。原始文件保持不变。
- 核心缓存：`data/processed/core/train.parquet`保留23个共有字段及标签、签收时间，`test.parquet`保留23个共有字段。43个节点不展开、不进入单包模型。
- 原始摘要、核心表摘要、转换异常和字段类型：`data/processed/core/preparation_manifest.json`、`schema.json`。
- 结构审计：`reports/audit_structure.json`，只包含决策时点共有字段；开发标签审计：`reports/audit_development.json`，已剔除898,846个最终留后包裹。
- 固定切分：`configs/splits.yaml`，键在`artifacts/splits/time_v1_20261117/`，来源及成熟度在`reports/splits/time_v1_20261117.json`。
- 校准队列成熟比例约94.7%–96.8%，并未完全成熟。独立留后隔离不消除已签收样本的选择偏差；M2不能预先假定优于未校准模型。

## 下载与来源

- 官方题包来源：<https://imupd-public.sheincorp.cn/d/external-contents/F4.zip>。
- 临时文件使用 `data/raw/F4.zip.part`；完整性校验通过后才改名为 `F4.zip`。
- 保留下载记录、文件字节数与SHA-256；解压前核对未压缩总大小及磁盘余量。
- 当前本机16GB内存，首版按20–25GiB项目预算准备，另保留15–20GiB系统余量。先做分块转换，模型与窗口串行运行。

从项目目录运行以下命令可下载或断点续传；脚本会记录进度，在下载完成后检查大小、必需文件、ZIP CRC和SHA-256，再解压并将 `.part` 改名。

```bash
bash scripts/download_official.sh
```

首次连接曾超时，可选环境变量 `F4_CDN_IP` 仅为单次下载指定已实测可用的官方CDN地址，不改系统DNS且保持证书校验；地址可能变化，不能永久依赖旧值。实际进度以 `reports/data_manifest.json`、`reports/downloads/F4.log` 及下载文件大小为准；后台任务PID见 `reports/downloads/F4.pid`，模板清单见 `reports/template_manifest.json`。下载脚本使用文件锁防止重复启动。

题包下载默认使用实时DNS。临时断网时记录 `waiting_for_network` 并退避重试，每次从文件当前大小续传，不使用会回退本轮输出的curl内部重试；证书或本地写入等错误会明确报错。依赖安装优先显式安装已校验的本地wheel，其余依赖使用项目内pip缓存并在网络失败后重试。

## Python环境

使用项目内 `.venv`，依赖安装成功后将实际版本保存到 `requirements.lock.txt`。

依赖和macOS的`libomp`已就绪。状态见`reports/environment_check.json`；依赖锁定采用`包名==版本`，本地wheel来源与SHA另存清单，避免本机绝对路径污染可移植依赖。

此前题包与依赖下载使用独立进程，历史记录在`reports/background_jobs.json`，该文件不是当前训练任务清单；下载已完成。后台下载脚本仅在运行期间使用`caffeinate -i`，没有设置登录或重启自启动。

```bash
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py
```

macOS缺少OpenMP时先安装`brew install libomp`。环境检查只用100行合成数据、5轮、单线程，不使用F4标签。

## 准备流程与冒烟

首次转换使用下列命令。现有核心缓存已生成，重复转换会拒绝覆盖；复核时可从结构审计继续。

```bash
.venv/bin/python scripts/prepare_data.py --raw-root data/raw/official --out data/processed/core
.venv/bin/python scripts/audit_data.py --mode structure --data-dir data/processed/core --out reports/audit_structure.json
.venv/bin/python scripts/make_splits.py --data-dir data/processed/core --config configs/splits.yaml --out artifacts/splits
.venv/bin/python scripts/audit_data.py --mode development --data-dir data/processed/core --exclude-ids artifacts/splits/time_v1_20261117/holdout_ids.parquet --out reports/audit_development.json
.venv/bin/python scripts/smoke_test.py
```

切分先只读键与付款/流入时间，锁定留后后才读取训练侧签收时间。配置和来源改变时不得覆盖既有切分。开发标签审计是回顾性描述，不能拿其事后统计声称所有参数在历史拟合时点均已可得。

只检查输入、参数与冻结来源，不训练、不评分：

```bash
.venv/bin/python scripts/backtest.py --config configs/experiments/baseline.yaml --fold dev_1 --dry-run --run-id YOUR_UNIQUE_PREFLIGHT_ID
```

`baseline.yaml`、`quantile.yaml`、`calibrated.yaml`分别对应B1/M1/M2，两个开发窗的预检报告在`reports/preparation/`。每次使用新`run_id`，拒绝覆盖已有报告。

所有实际配置为`stage: preparation`、`allow_real_training: false`。即使传入执行标志，真实回测也会在读训练数据前被拒绝。后续只有用户明确授权，才可调整阶段与开关；留后验收另需路线锁定。模型API可用，但本阶段仅在最多200行合成数据、每模型最多5轮、单线程下测试。

`evaluate.py`当前只允许合成小样例；`validate_submission.py`可独立检查键与预测格式。回测完成时按同一`run_id`关联模型、预测与评分，校验官方A/B一致性；pickle只加载可信本地工件。最终离线重训、正式测试预测与提交封装留待授权后的阶段。
