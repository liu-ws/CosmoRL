# PPO 环境、奖励与实现约定

[返回项目首页](../README.md)

当前实现采用终局奖励和完整 episode 的 GAE。

## episode

从初始尺度因子 `a_initial` 演化到终点 `a_final`，构成一个 episode。默认终点为 `1`。一个 action 控制一个 KDK 积分区间。训练会从不同初始条件反复开始。

## 输入状态

完整的环境状态包括粒子位置、动量、当前尺度因子、剩余预算等。

为了便于网络计算，先利用 CIC 将粒子的位置和动量网格化，构建密度场和动量场。
密度对比度输入为 `[B,1,N,N,N]`，CIC 动量输入为 `[B,3,N,N,N]`，三个分量依次对应网格的 $x,y,z$ 轴。编码器的每个卷积块同时传播标量场和极矢量场。转动或镜像盒子时，密度只随网格位置移动，动量还要按同一个正交矩阵变换其三个分量。

编码器使用中心点和六个周期面邻居。对邻居方向 $\mathbf n$，矢量更新使用邻居矢量 $\mathbf v$、沿该方向的投影 $\mathbf n(\mathbf n\cdot\mathbf v)$，以及由邻居密度特征产生的矢量 $\mathbf n s$。分别在六个方向聚合后，各项的可学习权重只混合特征通道，对三个空间分量共享。标量分支计算的 sigmoid 门同时缩放一个矢量的三个分量。该构造参考了 e3x 的方向滤波和标量门控，在固定立方网格上对整格周期平移、轴置换和坐标轴反号等变。[e3x 消息传递](https://e3x.readthedocs.io/stable/_modules/e3x/nn/modules.html#MessagePass) · [e3x 激活函数](https://e3x.readthedocs.io/stable/_modules/e3x/nn/activations.html)

经过两层卷积后，读出层对每个标量通道取空间平均，对每个矢量通道取局部平方模的空间平均。这样方向相反的区域在读出时都能贡献强度。将读出结果与当前尺度因子、剩余预算等拼接，交给策略头和价值头。

## 输出动作

选择下一段 $h_n=\Delta\ln a$，使用离散动作，例如候选集合

$$
\Delta\ln a \in \{0.01,\ 0.02,\ 0.04,\ 0.08\}.
$$

这些数值是第一版的默认动作，最终范围由稳定性和误差实验确定。包含中间检查点时，推进终点为

$$
a_{n+1}=\min\{a_n e^{h_n},\ a_{\rm next\ checkpoint},\ a_{\rm final}\}.
$$

当前环境使用终点作为唯一检查点，并在终点给予奖励。

距离检查点很近时，选择不同动作的效果可能是一样的，可合并重复动作，避免无意义的动作熵。

数值 NaN、无合法动作、预算耗尽但未到终点均判定为失败。

## 计算预算

预算 $B$ 定义为**初始化求解之后允许新增的 PM 求解次数**。初始化仍计入总运行时间并单独报告。

只检查“下一步花费是否小于剩余预算”不够：策略可能前期全选小步，最后无力到达 $a=1$。

令最大允许步长为 $h_{\max}$。无中间检查点时，最少剩余推进步数为

$$
C_{\min}(a)=\left\lceil\frac{\ln(a_{\rm final}/a)}{h_{\max}}\right\rceil
$$

有多个必须落点的检查点时，对后续每个区间分别取 ceiling 后求和。实现时处理浮点误差，避免临近终点时多算一步。一个候选动作只有满足

$$
\boxed{1+C_{\min}(a_{n+1})\le B_{\rm remaining}}
$$

才合法，也就是让下一步还有完成任务的可能。

动作 mask 要同时用于采样、旧对数概率的保存和更新时的新概率计算。rollout 保存原始 mask，不能用别的状态重新生成。训练与评估使用同一套预算规则。

## 科学误差与奖励

### 不只看功率谱

功率谱相同不代表结构出现在相同位置。因此至少同时评估功率谱与密度场的一致性。定义固定波数带通算子 $\mathcal{B}$，比较相同网格上的场：

$$
e_\delta=
\frac{\|\mathcal B\delta-\mathcal B\delta_{\rm ref}\|_2^2}
{\|\mathcal B\delta_{\rm ref}\|_2^2+\epsilon_\delta},
$$

$$
e_P=\frac1{N_k}\sum_{k\in\mathcal K}
\left[\frac{P(k)-P_{\rm ref}(k)}{P_{\rm ref}(k)+\epsilon_P}\right]^2,
$$

$$
r(k)=\frac{\operatorname{Re}P_{\rm cross}(k)}
{\sqrt{P(k)P_{\rm ref}(k)}}.
$$

最后一个 $r(k)$ 是交叉相关系数。有效模式、$\epsilon$、分箱和过滤方式在验证阶段固定。

用粒子数与网格共同限定可信波数。可以先考察 $k_{\rm max}\le0.3\min(k_{\rm Nyquist,grid},k_{\rm Nyquist,particle})$，但这只是保守起点；最终范围由分辨率测试确认。所有方法使用相同质量分配与谱估计方法。FastPM 使用功率谱与交叉相关作为重要比较量，可参考其评估思路而不是照搬其精度结论。[参考：FastPM][fastpm]

### MVP 奖励：固定预算下的终局误差

先把原始误差转换成有界损失：

$$
\mathcal L(a)=w_\delta\frac{e_\delta}{\tau_\delta+e_\delta}
+w_P\frac{e_P}{\tau_P+e_P},
\qquad w_\delta,w_P\ge0,\quad w_\delta+w_P=1,\quad \tau_\delta,\tau_P>0.
$$

$\tau$ 是根据科学容限或训练/验证集确定的尺度。

只在成功到达最终尺度因子时给奖励：

$$
r_n=0\quad\text{（中间步）},\qquad
r_{\rm terminal}=-\mathcal L(a_{\rm final}).
$$

合法完成的总回报在 $[-1,0]$ 内；失败终止取 $-2$。这样提前崩溃不会比合法完成更有利。

采用有限 episode 的 $\gamma=1$。动作改变了物理时间跨度和决策次数，直接套用每步 $\gamma=0.99$ 会使同一终局误差因为用了不同步数而受到不同折扣。当前目标是固定终点、固定预算的最终精度，不需要这种偏好。

### 可选的密集奖励

若终局奖励学习困难，可以在固定尺度因子检查点使用

$$
r_j=\mathcal L(a_{j-1})-\mathcal L(a_j),
$$

其他步给 0，**不再额外叠加相同的终局损失**。$\gamma=1$ 时总回报为 $\mathcal L(a_{\rm initial})-\mathcal L(a_{\rm final})$，保留最终误差目标。

这里的差值不是“本动作相对不采取动作的因果收益”：系统本来就在演化。误差增长导致负奖励完全正常。它只是对同一终局目标的另一种奖励记账方式。

## GAE 与 PPO 目标

记第 $n$ 步的输入状态为 $o_n$，离散动作索引为 $j_n$。PPO 的重要性采样比率为

$$
q_n(\theta)= \exp\left[ \log\pi_\theta(j_n\mid o_n)-\log\pi_{\rm old}(j_n\mid o_n) \right].
$$

策略目标为

$$
J_{\rm clip}=\mathbb E_n\left[ \min\left(q_n\hat A_n, \operatorname{clip}(q_n,1-\epsilon,1+\epsilon)\hat A_n\right) \right].
$$

以下 $s_n$ 表示完整环境状态，$o_n$ 表示策略观测。为了优化该目标，需要估计优势函数 $A^\pi(s_n, j_n) = Q^\pi(s_n,j_n)-V^\pi(s_n)$。在实际训练中，状态价值 $V^\pi(s_n)$ 由神经网络价值头 $V_\phi(s_n)$ 近似输出；而动作价值 $Q^\pi(s_n, j_n)$ 的估计则面临**偏差与方差的权衡**：

### 1. 蒙特卡洛（MC）点估计（回报估计无偏但通常方差较高）

最直接的方式是利用整条采样轨迹的实际折扣回报 $G_n = \sum_{l=0}^{\infty}\gamma^l r_{n+l}$ 作为期望的单点估计：

$$
Q^\pi(s_n, j_n) = \mathbb{E}[G_n\mid s_n,j_n] \approx G_n \implies \hat A_n^{\rm MC} = G_n - V_\phi(s_n).
$$

这种回报估计虽然无偏，但 $G_n$ 累加了从 $n$ 时刻直到轨迹结束的所有随机奖励和状态转移噪声，步数越长通常方差越大，极易导致策略梯度更新震荡。

### 2. 单步时序差分（1-step TD）估计（通常方差较低但可能有偏差）

为了降低后续整条轨迹带来的高方差，可以利用贝尔曼方程将 $G_n$ 仅展开一步：

$$
\begin{aligned}
Q^\pi(s,j) &=\mathbb E[r_n+\gamma G_{n+1}\mid s_n=s,j_n=j]\\
&=\mathbb E[r_n\mid s,j] +\gamma\mathbb E_{s_{n+1}\mid s_n=s,j_n=j}[V^\pi(s_{n+1})]\\
&=\mathbb E[r_n+\gamma V^\pi(s_{n+1})\mid s_n=s,j_n=j]
\end{aligned}
$$

用单步实际采样 $(r_n, s_{n+1})$ 替代外层期望，并用价值网络 $V_\phi$ 替代真实的 $V^\pi$（即自举 Bootstrap），便构造出**一步优势估计**，即单步 TD 误差 $\delta_n$：

$$
\hat A_n^{(1)} = \delta_n = r_n+\gamma V_\phi(s_{n+1})-V_\phi(s_n).
$$

$\hat A_n^{(1)}$ 只包含一步真实采样奖励 $r_n$，其余未来回报全部采用价值网络输出的平均值估计 $V_\phi(s_{n+1})$ 截断，因此通常方差较低；但当价值网络 $V_\phi$ 尚未收敛准确时，会引入较大的估计偏差（Bias）。

### 3. 多步优势估计与 GAE（折中偏差与方差）

既然展开 1 步偏差大、展开到底（$\infty$ 步）方差大，自然的推广是向前展开 $k$ 步真实奖励，在第 $n+k$ 步再用 $V_\phi(s_{n+k})$ 截断，构造 **$k$-步优势估计** $\hat A_n^{(k)}$。利用裂项相消（插入并减去中间时刻的价值项 $\gamma^l V_\phi(s_{n+l})$），$\hat A_n^{(k)}$ 恰好可以表示为前 $k$ 步 TD 误差的折扣累加：

$$
\begin{aligned}
\hat A_n^{(1)} &= r_n + \gamma V_\phi(s_{n+1}) - V_\phi(s_n) = \delta_n \\
\hat A_n^{(2)} &= r_n + \gamma r_{n+1} + \gamma^2 V_\phi(s_{n+2}) - V_\phi(s_n) \\
&= \big(r_n + \gamma V_\phi(s_{n+1}) - V_\phi(s_n)\big) + \gamma\big(r_{n+1} + \gamma V_\phi(s_{n+2}) - V_\phi(s_{n+1})\big) = \delta_n + \gamma\delta_{n+1} \\
&\;\;\vdots \\
\hat A_n^{(k)} &= \sum_{l=0}^{k-1}\gamma^l r_{n+l} + \gamma^k V_\phi(s_{n+k}) - V_\phi(s_n) = \sum_{l=0}^{k-1}\gamma^l \delta_{n+l}.
\end{aligned}
$$

为了避免手动选择固定的截断步数 $k$，GAE（广义优势估计）引入衰减因子 $\lambda \in [0, 1]$，对所有步数的优势估计 $\hat A_n^{(k)}$ 进行**指数加权移动平均**（权重为 $(1-\lambda)\lambda^{k-1}$，在 $0\le\lambda<1$ 时满足归一化条件 $\sum_{k=1}^\infty (1-\lambda)\lambda^{k-1} = 1$）：

$$
\begin{aligned}
\hat A_n^{\rm GAE} &= (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} \hat A_n^{(k)} \\
&= (1-\lambda)\Big( \delta_n + \lambda(\delta_n + \gamma\delta_{n+1}) + \lambda^2(\delta_n + \gamma\delta_{n+1} + \gamma^2\delta_{n+2}) + \cdots \Big) \\
&= (1-\lambda)\Big( \delta_n(1+\lambda+\lambda^2+\cdots) + \gamma\delta_{n+1}(\lambda+\lambda^2+\cdots) + \cdots \Big) \\
&= (1-\lambda)\sum_{k=0}^{\infty} \gamma^k \delta_{n+k} \frac{\lambda^k}{1-\lambda} \\
&= \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{n+k}.
\end{aligned}
$$

对于有限长度 $T$ 的采样轨迹，截断至终止步 $T-1$ 即得到最终的 GAE 计算公式（真正终止时取 $V_\phi(s_T)=0$；仅因 rollout 长度截断时仍需自举）：

$$
\boxed{ \hat A_n^{\rm GAE} =\sum_{k=0}^{T-1-n}(\gamma\lambda)^k\delta_{n+k} }.
$$

- **当 $\lambda=0$ 时**：$\hat A_n^{\rm GAE} = \delta_n$，退化为单步 TD 估计（通常方差较低，偏差取决于价值估计误差）。
- **当 $\lambda=1$ 时**：中间所有的 $V_\phi(s_{n+k})$ 完全裂项相消，$\hat A_n^{\rm GAE} = G_n - V_\phi(s_n)$，在真正终止时退化为蒙特卡洛单次采样估计（回报估计无偏，通常方差较高）。
- **当 $0 < \lambda < 1$ 时**：中间状态价值 $V_\phi(s_{n+k})$ 不会完全抵消，而是以 $(1-\lambda)$ 的比例保留在每一项中，实现了近期奖励信真实采样、远期回报信价值网络均值预测的平滑过渡。

在代码实现时，利用展开式的递推结构，等价地从轨迹末尾倒序计算即可在 $O(T)$ 时间内完成求解：

$$
\hat A_n=\delta_n+\gamma\lambda\hat A_{n+1},\qquad \hat A_T=0.
$$

参考：[GAE 原论文][gae]。

[fastpm]: https://academic.oup.com/mnras/article/463/3/2273/2646558 "Feng et al. (2016), FastPM: a new scheme for fast simulations of dark matter and haloes"
[ppo]: https://arxiv.org/abs/1707.06347 "Schulman et al. (2017), Proximal Policy Optimization Algorithms"
[gae]: https://arxiv.org/abs/1506.02438 "Schulman et al. (2015), High-Dimensional Continuous Control Using Generalized Advantage Estimation"
