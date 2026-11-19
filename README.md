# CosmoRL

用 CLASS 生成宇宙学初值，用 PyTorch 实现周期性 Particle-Mesh（PM）引力模拟，并用 PPO 选择积分步长。

> [!IMPORTANT]
> 本项目为宇宙学课程大作业，包含大量近似，可能存在未知错误。

## 快速开始

在 Linux 或 WSL2 下需要 uv、`make` 和 GCC。

```sh
uv sync --locked
uv run --locked main.py simulate
uv run --locked main.py train
uv run --locked main.py evaluate policy.pt
```

## 公式与设计

- [宇宙学公式推导](docs/cosmology.md)
- [PM 方法推导](docs/pm_solver.md)
- [PPO 设计与推导](docs/rl.md)
