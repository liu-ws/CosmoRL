# CosmoRL

用 CLASS 生成平直 $\Lambda\mathrm{CDM}$ 宇宙学初值，用 PyTorch 实现周期性 Particle-Mesh（PM）引力模拟，并基于 $O(3)$ 等变三维卷积网络与带动作掩码的 PPO 学习自适应积分步长策略。

> [!IMPORTANT]
> 本项目为宇宙学课程大作业，包含若干物理与数值近似，仅供学习与实验研究参考。

## 目录结构

```text
CosmoRL/
├── configs/                 # TOML 实验配置文件模板
│   ├── default.toml         # 完整默认配置
│   ├── simulate.toml        # 固定步长 PM 模拟配置
│   ├── train.toml           # PPO 自适应步长训练配置
│   └── evaluate.toml        # 策略评估配置
├── docs/                    # 物理原理、数值方法与强化学习推导文档
│   ├── cosmology.md         # 宇宙学背景、运动方程、线性增长与 1LPT 初值
│   ├── pm_solver.md         # Particle-Mesh 引力求解与 KDK 积分
│   └── rl.md                # 自适应步长 MDP、O(3) 等变网络与 PPO/GAE
├── src/cosmorl/             # 核心 Python 包
│   ├── config.py            # 强类型配置体系与 TOML/JSON 加载校验
│   ├── cli.py               # 配置驱动命令行工具入口
│   ├── workflows.py         # 高层实验编排 API（simulate / train / evaluate）
│   ├── physics/             # 宇宙学背景、CLASS 初值与 PM 求解器
│   │   ├── cosmology.py
│   │   └── pm.py
│   └── rl/                  # MDP 环境、等变网络与 PPO 算法
│       ├── environment.py
│       ├── network.py
│       └── ppo.py
├── tests/                   # 分模块单元测试
└── pyproject.toml
```

## 安装与快速开始

在 Linux 或 WSL2 环境下需要 `uv`（或 Python 3.12+）、`make` 与 GCC（用于编译 CLASS）：

```sh
uv sync
```

所有实验参数均通过配置文件（TOML 或 JSON）管理，安装后可直接使用 `cosmorl` 命令（或 `python -m cosmorl`）：

```sh
# 1. 生成一份带注释的默认配置模板
cosmorl init-config configs/my_exp.toml

# 2. 校验配置文件合法性并查看摘要
cosmorl validate-config -c configs/train.toml

# 3. 运行固定步长 PM 模拟
cosmorl simulate -c configs/simulate.toml

# 4. 训练 PPO 自适应步长策略
cosmorl train -c configs/train.toml

# 5. 对比评估策略与同预算固定步长基线
cosmorl evaluate -c configs/evaluate.toml
```

## Python API 示例

```python
from cosmorl import load_config, run_evaluation, run_simulation, run_training

config = load_config("configs/train.toml")
sim_result = run_simulation(config)
train_result = run_training(config)
eval_result = run_evaluation(load_config("configs/evaluate.toml"))
```

## 理论推导与实验方案

- [宇宙学背景、粒子运动与线性增长](docs/cosmology.md)
- [Particle-Mesh 引力求解与 KDK 积分](docs/pm_solver.md)
- [自适应步长强化学习、$O(3)$ 等变网络与 PPO 推导](docs/rl.md)
