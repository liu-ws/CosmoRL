# Particle-Mesh 引力求解与 KDK 积分

本文给出三维周期性立方盒中无碰撞物质 Particle-Mesh（PM）引力求解器的完整数学原理与数值方案，涵盖 Cloud-In-Cell（CIC）质量分配、基于离散傅里叶变换（DFT）的连续谱力核、Nyquist 平面实数性处理、零自力与总内力守恒证明，以及与 Kick–Drift–Kick（KDK）积分器的耦合。

## 1. 方法动机与复杂度

直接对 $N_p$ 个粒子两两计算万有引力（含周期镜像求和）的时间复杂度为 $O(N_p^2)$，在大粒子数下代价高昂。Particle-Mesh 方法通过规则三维网格中介引力计算，将单步力求解分为四步：

1. **质量分配**：将离散粒子按空间核函数平滑分配至每轴 $N$ 点（总网格数 $N_g = N^3$）的均匀规则网格，构造密度对比度 $\delta$；
2. **谱空间求解 Poisson 方程**：经快速傅里叶变换（FFT）将偏微分方程化为代数乘法，求得傅里叶空间引力场；
3. **逆变换恢复网格力**：经逆 FFT 获得实空间各网格点上的三分量引力场；
4. **力插值**：用与质量分配相同的核函数将网格力插值回每个粒子的连续位置。

单次 PM 力求解的总计算复杂度为：

$$
O(N_p) + O(N_g \log N_g).
$$

其代价是空间分辨率受限于网格间距 $\Delta = 1/N$，无法精确解析远小于单网格尺度的近距离两体引力。下文统一记粒子编号为 $p \in \{1,\dots,N_p\}$、三维网格索引为 $\mathbf j = (j_x, j_y, j_z) \in \{0,\dots,N-1\}^3$、离散傅里叶模式索引为 $\mathbf n = (n_x, n_y, n_z)$。

## 2. 无量纲目标方程与离散化约定

由宇宙学背景与共动坐标推导（参见 `cosmology.md`），以共动盒长 $L$ 与今日 Hubble 常数 $H_0$ 无量纲化后，粒子在单位周期盒 $[0,1)^3$ 内的运动方程与 Poisson 方程为：

$$
\boxed{
\frac{d\mathbf y}{da} = \frac{\mathbf u}{a^3\mathcal E(a)}, \qquad
\frac{d\mathbf u}{da} = \frac{\mathbf F}{a^2\mathcal E(a)}, \qquad
\mathbf F = -\nabla_y\psi, \qquad
\nabla_y^2\psi = C\delta, \quad C \equiv \frac{3}{2}\Omega_m.
}
$$

网格与场的离散化约定如下：

| 符号 | 物理含义 | 离散化约定 |
| --- | --- | --- |
| $\mathbf y_p$ | 粒子无量纲共动坐标 | 各分量经周期折回 $y \leftarrow y - \lfloor y \rfloor$ 位于 $[0,1)$ |
| $N_p, N$ | 粒子总数与每轴网格点数 | 正整数，网格间距 $\Delta = 1/N$（物理共动间距 $L/N$） |
| $\mathbf y_{\mathbf j} = \mathbf j/N$ | 网格节点位置 | $j_x, j_y, j_z \in \{0,\dots,N-1\}$ |
| $\delta_{\mathbf j}$ | 网格密度对比度 | 定义于同位网格节点 $\mathbf y_{\mathbf j}$ 上的无量纲标量场 |
| $\mathbf F_{\mathbf j}$ | 网格引力场 | 定义于同一套节点 $\mathbf y_{\mathbf j}$ 上的三分量实矢量场 |
| $\mathbf F_p$ | 粒子所受引力 | 与粒子 $p$ 一一对应的三分量实矢量 |

## 3. Cloud-In-Cell（CIC）质量分配

### 3.1 一维线性分配与三维张量积权重

粒子位置 $\mathbf y_p$ 一般不落在网格节点上。在一维情形下，定义网格单位坐标 $s = Ny$，其下邻节点索引与小数偏移分别为：

$$
i = \lfloor s \rfloor \in \{0,\dots,N-1\}, \qquad f = s - i \in [0, 1).
$$

CIC 采用一维线性（分段三角核）权重将粒子分配给相邻两节点 $i$ 与 $(i+1)\bmod N$：

$$
w_0(f) = 1 - f, \qquad w_1(f) = f.
$$

三维 CIC 权重为三个坐标轴一维权重的张量积。令 $\mathbf i = \lfloor N\mathbf y_p \rfloor$、$\mathbf f = N\mathbf y_p - \mathbf i$，对八个角点偏移 $\mathbf b = (b_x, b_y, b_z) \in \{0,1\}^3$，相邻网格节点索引与对应权重为：

$$
\mathbf j(\mathbf b) = (\mathbf i + \mathbf b) \bmod N,
$$

$$
\boxed{W_{p,\mathbf b} = w_{b_x}(f_x)\,w_{b_y}(f_y)\,w_{b_z}(f_z).}
$$

对八个角点求和：

$$
\sum_{\mathbf b \in \{0,1\}^3} W_{p,\mathbf b}
= \big[(1-f_x)+f_x\big]\big[(1-f_y)+f_y\big]\big[(1-f_z)+f_z\big] = 1,
$$

因此 CIC 分配严格保持总粒子数与总质量守恒。

### 3.2 由粒子计数构造密度对比度

记 $W_{\mathbf j}(\mathbf y_p)$ 为粒子 $p$ 贡献给网格点 $\mathbf j$ 的 CIC 权重（在八个相邻角点之外为零），累加全盒粒子得到网格有效粒子数：

$$
c_{\mathbf j} = \sum_{p=1}^{N_p} W_{\mathbf j}(\mathbf y_p), \qquad \sum_{\mathbf j} c_{\mathbf j} = N_p.
$$

对于等质量粒子（单粒子质量 $m_p$），单个网格单元的共动体积为 $(L/N)^3$，全盒共动体积为 $L^3$，故局部共动密度与平均共动密度分别为：

$$
\rho_{{\rm com},\mathbf j} = \frac{m_p c_{\mathbf j}}{(L/N)^3} = \frac{m_p N^3}{L^3}c_{\mathbf j}, \qquad
\bar\rho_{\rm com} = \frac{m_p N_p}{L^3}.
$$

由于物理密度与共动密度仅差公因子 $a^{-3}$，在比值中完全抵消，从而得到网格密度对比度：

$$
\boxed{\delta_{\mathbf j} = \frac{\rho_{{\rm com},\mathbf j}}{\bar\rho_{\rm com}} - 1 = \frac{N^3}{N_p}c_{\mathbf j} - 1.}
$$

上式在精确算术下满足全盒零均值条件 $\sum_{\mathbf j}\delta_{\mathbf j} = \frac{N^3}{N_p}N_p - N^3 = 0$。

## 4. 傅里叶空间求解周期 Poisson 方程

### 4.1 连续周期盒的谱力核

单位周期盒 $[0,1)^3$ 上的正交平面波基函数为 $e^{i\mathbf q\cdot\mathbf y}$，其中无量纲波矢为：

$$
\mathbf q = 2\pi\mathbf n, \qquad \mathbf n \in \mathbb Z^3.
$$

物理共动波矢为 $\mathbf k = \mathbf q/L$。在平面波基下，空间微分算子转化为代数乘法：

$$
\nabla_y e^{i\mathbf q\cdot\mathbf y} = i\mathbf q\,e^{i\mathbf q\cdot\mathbf y}, \qquad
\nabla_y^2 e^{i\mathbf q\cdot\mathbf y} = -q^2 e^{i\mathbf q\cdot\mathbf y},
$$

其中 $q^2 = q_x^2 + q_y^2 + q_z^2$。将其代入 $\nabla_y^2\psi = C\delta$ 与 $\mathbf F = -\nabla_y\psi$，对任意非零波矢 $\mathbf q \neq \mathbf 0$ 逐模解得：

$$
-q^2\widetilde\psi(\mathbf q) = C\widetilde\delta(\mathbf q)
\;\Longrightarrow\;
\widetilde\psi(\mathbf q) = -\frac{C}{q^2}\widetilde\delta(\mathbf q),
$$

$$
\widetilde F_\alpha(\mathbf q) = -i q_\alpha\widetilde\psi(\mathbf q)
= \boxed{iC\frac{q_\alpha}{q^2}\widetilde\delta(\mathbf q), \qquad \alpha \in \{x, y, z\}, \quad q^2 > 0.}
$$

### 4.2 零模相容性条件

对周期 Poisson 方程在单位盒内积分，由散度定理及周期边界通量抵消，有：

$$
\int_{[0,1)^3} \nabla_y^2\psi\,d^3y = 0 = C\int_{[0,1)^3} \delta\,d^3y.
$$

零模（$\mathbf q = \mathbf 0$）对应空间均匀背景，引力势的任意加性常数不影响力场，故规定：

$$
\widetilde\psi(\mathbf 0) = 0, \qquad \widetilde{\mathbf F}(\mathbf 0) = \mathbf 0.
$$

## 5. 离散傅里叶变换与 Nyquist 模式处理

### 5.1 DFT 归一化与 Hermitian 共轭对称性

在 $N\times N\times N$ 均匀网格上，采用标准三维离散傅里叶正反变换对：

$$
\widehat\delta_{\mathbf n} = \sum_{j_x,j_y,j_z=0}^{N-1} \delta_{\mathbf j}\exp\!\left(-\frac{2\pi i}{N}\mathbf n\cdot\mathbf j\right), \qquad
\delta_{\mathbf j} = \frac{1}{N^3}\sum_{\mathbf n} \widehat\delta_{\mathbf n}\exp\!\left(+\frac{2\pi i}{N}\mathbf n\cdot\mathbf j\right).
$$

由于 $\delta_{\mathbf j}$ 为实数场，其离散傅里叶系数满足 Hermitian 共轭对称性：

$$
\widehat\delta_{-\mathbf n \bmod N} = \widehat\delta_{\mathbf n}^*.
$$

因此，实数场的全部频谱信息仅由最后一个坐标轴的非负频率半谱 $i_z \in \{0, 1, \dots, \lfloor N/2\rfloor\}$（共 $N^2(\lfloor N/2\rfloor + 1)$ 个复数模式）唯一确定，其余模式由共轭对称性自动补全。

前两个完整频率轴的数组下标 $i \in \{0,\dots,N-1\}$ 到有符号整数波数 $n(i)$ 的映射为：

$$
n(i) = \begin{cases}
i, & 0 \le i \le \lfloor N/2 \rfloor, \\
i - N, & i > \lfloor N/2 \rfloor,
\end{cases}
$$

半谱轴直接取 $n_z = i_z \in \{0,\dots,\lfloor N/2\rfloor\}$，进而确定无量纲波矢 $\mathbf q = 2\pi(n_x, n_y, n_z)$。

### 5.2 偶数网格 Nyquist 平面的实数性证明与修正

当 $N$ 为偶数时，任意一轴在索引 $i_\alpha = N/2$ 处对应的 Nyquist 模式满足 $n_\alpha = N/2 \equiv -N/2 \pmod N$。考虑该平面上的模式 $\mathbf n$ 与其共轭配对模式 $-\mathbf n \bmod N$：
- 沿 Nyquist 轴分量不变：$n_\alpha \equiv -n_\alpha \pmod N$，对应平面波 $e^{i\pi j_\alpha} = (-1)^{j_\alpha}$ 为纯实函数；
- 若此时其余两轴均为零或 Nyquist 频率，则 $\mathbf n \equiv -\mathbf n \pmod N$，由 Hermitian 对称性 $\widehat\delta_{\mathbf n} = \widehat\delta_{\mathbf n}^*$ 知 $\widehat\delta_{\mathbf n}$ 必为实数；若乘以非零纯虚系数 $i q_\alpha$，则 $\widehat F_{\alpha,\mathbf n}$ 变为纯虚数，违背实数场的共轭对称条件；
- 更一般地，对任意位于 $i_\alpha = N/2$ 平面上的混合模式 $\mathbf n = (N/2, n_y, n_z)$，其共轭模式为 $\mathbf n' = (N/2, -n_y, -n_z)$，满足 $q_\alpha(\mathbf n') = +q_\alpha(\mathbf n)$ 而非 $-q_\alpha(\mathbf n)$。因此：

$$
\left(i C \frac{q_\alpha(\mathbf n)}{q^2}\widehat\delta_{\mathbf n}\right)^*
= -i C \frac{q_\alpha(\mathbf n)}{q^2}\widehat\delta_{\mathbf n}^*
= -i C \frac{q_\alpha(\mathbf n')}{q^2}\widehat\delta_{\mathbf n'}
\neq \widehat F_{\alpha,\mathbf n'},
$$

除非令求导方向的分子波数在该平面上为零！而在其余非 Nyquist 方向 $\beta \neq \alpha$ 上，$q_\beta(\mathbf n') = -q_\beta(\mathbf n)$ 依然成立，Hermitian 对称性自动保持。

因此，定义修正后的求导波数：

$$
q_\alpha^{(D)} = \begin{cases}
0, & N \text{ 为偶数且 } i_\alpha = N/2, \\
q_\alpha, & \text{其他情况},
\end{cases}
$$

最终离散谱力核为：

$$
\boxed{
\widehat F_{\alpha,\mathbf n} = iC\frac{q_\alpha^{(D)}}{q_x^2 + q_y^2 + q_z^2}\widehat\delta_{\mathbf n},
\qquad \mathbf q \neq \mathbf 0, \qquad \widehat{\mathbf F}_{\mathbf 0} = \mathbf 0.
}
$$

注意**分母始终保留完整的 $q^2 = q_x^2 + q_y^2 + q_z^2$，仅对当前求导方向的分子置零**。这样既严格保证逆变换后的力场为纯实数，又保留了混合 Nyquist 模式在横向方向上的准确引力。当 $N$ 为奇数时不存在 $i_\alpha = N/2$ 平面，无需此修正。

### 5.3 解析检验：单模余弦密度波

为验证引力符号、归一化与混合 Nyquist 模式的正确性，可构造解析测试场：

1. **奇数网格纯单模**：取 $\delta(\mathbf y) = \cos(2\pi y_x) = \frac{1}{2}(e^{i 2\pi y_x} + e^{-i 2\pi y_x})$，仅含 $q_x = \pm 2\pi$ 两模式。代入 $\widetilde F_x = iC\frac{q_x}{q^2}\widetilde\delta$，得：

   $$
   F_x(\mathbf y) = \frac{1}{2}\left(iC\frac{2\pi}{(2\pi)^2}e^{i2\pi y_x} - iC\frac{2\pi}{(2\pi)^2}e^{-i2\pi y_x}\right)
   = -\frac{C}{2\pi}\sin(2\pi y_x)
   = -\frac{3\Omega_m}{4\pi}\sin(2\pi y_x),
   $$

   且 $F_y = F_z = 0$。在过密峰 $y_x = 0$ 两侧，$y_x \in (0, 1/2)$ 处 $F_x < 0$ 指向峰值，符号符合引力吸引。

2. **偶数网格混合 Nyquist 模**：取 $N=8$ 与 $\delta(\mathbf y) = \cos(\pi N y_x)\cos(2\pi y_y)$，此时 $n_x = 4$（Nyquist）、$n_y = \pm 1$、$n_z = 0$，$q^2 = (2\pi)^2(4^2 + 1^2) = 17(2\pi)^2$。由修正核知 $F_x = F_z = 0$，而 $y$ 方向保留横向引力：

   $$
   F_y(\mathbf y) = -\frac{3\Omega_m}{34\pi}\cos(\pi N y_x)\sin(2\pi y_y).
   $$

## 6. 力插值与动量守恒证明

经三维逆 DFT 获得网格引力场 $\mathbf F_{\mathbf j}$ 后，使用与质量分配完全相同的八个 CIC 权重将其插值回粒子位置：

$$
\boxed{\mathbf F_p = \sum_{\mathbf b \in \{0,1\}^3} W_{p,\mathbf b}\,\mathbf F_{\mathbf j(\mathbf b)}.}
$$

**定理（零自力与牛顿第三定律）**：当质量分配与力插值采用相同的 CIC 权重，且使用上述连续谱力核时，在精确算术下每个粒子所受的自身网格引力为零，且系统总内力精确为零。

**证明**：将全盒 $N_p$ 个粒子到 $N_g$ 个网格点的分配写成矩阵 $A \in \mathbb R^{N_g \times N_p}$，其元素 $A_{\mathbf j, p} = W_{\mathbf j}(\mathbf y_p)$。则网格密度对比度向量满足 $\delta = \frac{N_g}{N_p}A\mathbf 1 - \mathbf 1$，而任意固定方向 $\alpha$ 的粒子力插值矩阵恰好为 $A^\top$，即 $\mathbf F_\alpha^{\rm part} = A^\top \mathbf F_\alpha^{\rm grid}$。

考察方向 $\alpha$ 的谱力算子 $K_\alpha = \mathcal F^{-1}\operatorname{diag}\!\left(iC\frac{q_\alpha^{(D)}}{q^2}\right)\mathcal F$。由于乘子 $g_\alpha(\mathbf n) = iC\frac{q_\alpha^{(D)}}{q^2}$ 满足：
- 共轭对称性 $g_\alpha(-\mathbf n) = g_\alpha(\mathbf n)^*$（保证 $K_\alpha$ 为实矩阵）；
- 奇函数性质 $g_\alpha(-\mathbf n) = -g_\alpha(\mathbf n)$（因 $q_\alpha^{(D)}$ 在非 Nyquist 处为奇函数，在 Nyquist 处为零）；

由实空间卷积核与频谱的关系，矩阵元素 $(K_\alpha)_{\mathbf j, \mathbf l}$ 仅依赖于周期位移 $\mathbf j - \mathbf l$，且：

$$
(K_\alpha)_{\mathbf l, \mathbf j}
= \frac{1}{N^3}\sum_{\mathbf n} g_\alpha(\mathbf n) e^{i\frac{2\pi}{N}\mathbf n\cdot(\mathbf l - \mathbf j)}
= \frac{1}{N^3}\sum_{\mathbf n} g_\alpha(-\mathbf n) e^{i\frac{2\pi}{N}\mathbf n\cdot(\mathbf j - \mathbf l)}
= -(K_\alpha)_{\mathbf j, \mathbf l},
$$

即网格力算子是**反对称矩阵**：$K_\alpha^\top = -K_\alpha$。此外常数向量 $\mathbf 1$ 位于 $K_\alpha$ 的零空间（$K_\alpha\mathbf 1 = \mathbf 0$，对应零模置零），故粒子力可写为：

$$
\mathbf F_\alpha^{\rm part} = \frac{N_g}{N_p} A^\top K_\alpha A\,\mathbf 1 = \frac{N_g}{N_p} M_\alpha \mathbf 1, \qquad M_\alpha \equiv A^\top K_\alpha A.
$$

验证有效粒子间相互作用矩阵 $M_\alpha$ 的转置：

$$
M_\alpha^\top = (A^\top K_\alpha A)^\top = A^\top K_\alpha^\top A = -A^\top K_\alpha A = -M_\alpha.
$$

由于 $M_\alpha$ 反对称，立刻推出两条物理守恒律：
1. **无自引力**：所有对角元 $(M_\alpha)_{pp} = 0$，单个粒子不会对自己施加虚假网格力；
2. **总内力为零（动量守恒）**：任意两粒子间满足 $(M_\alpha)_{pq} = -(M_\alpha)_{qp}$，故全系统合力

$$
\sum_{p=1}^{N_p} F_{\alpha, p}^{\rm part} = \frac{N_g}{N_p}\mathbf 1^\top M_\alpha \mathbf 1 = 0.
$$

## 7. 与 KDK 积分器的耦合

结合第 2 节的运动方程与 `cosmology.md` 中的漂移因子 $\mathcal D$、踢动因子 $\mathcal K$，单步推进 $a_n \to a_{n+1}$（中点 $a_m = \frac{1}{2}(a_n + a_{n+1})$）按 Kick–Drift–Kick 顺序执行：

$$
\begin{aligned}
\mathbf F_n &= \operatorname{PM}(\mathbf y_n), \\
\mathbf u_{n+1/2} &= \mathbf u_n + \mathcal K(a_n, a_m)\mathbf F_n, \\
\mathbf y_{n+1} &= \operatorname{wrap}\!\big[\mathbf y_n + \mathcal D(a_n, a_{n+1})\mathbf u_{n+1/2}\big], \\
\mathbf F_{n+1} &= \operatorname{PM}(\mathbf y_{n+1}), \\
\mathbf u_{n+1} &= \mathbf u_{n+1/2} + \mathcal K(a_m, a_{n+1})\mathbf F_{n+1}.
\end{aligned}
$$

该格式具有两项关键数值性质：
- **时间反演对称与二阶精度**：半步踢动—整步漂移—半步踢动的对称分裂保持哈密顿相空间体积（保辛性），在平滑步长下局部截断误差为 $O(\Delta a^3)$、全局误差为 $O(\Delta a^2)$；
- **FSAL（First Same As Last）力复用**：第 $n$ 步末尾计算的力 $\mathbf F_{n+1}$ 与第 $n+1$ 步开头所需的力完全相同。因此除初始时刻 $a_i$ 需要 1 次初始化 PM 求解外，此后每推进 1 个 KDK 步仅新增 1 次 PM 引力计算。
