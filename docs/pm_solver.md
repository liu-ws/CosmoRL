# Particle-Mesh 引力求解

本文解释 PM 求解器怎样实现。目标是周期性立方盒中的无碰撞物质，使用等质量粒子、CIC 和 连续 Fourier 力核。

## 1. 方法动机

直接计算每一对粒子之间的引力需要约 $O(N_p^2)$ 次操作，周期边界还意味着不能只考虑盒内的一份粒子。PM 把离散粒子的质量平滑到规则网格上，在周期网格上解 Poisson 方程，再把网格上的力插值回粒子。

设粒子总数为 $N_p$，每个方向有 $N$ 个网格点，总网格点数为 $N_g=N^3$。一次 PM 力求解的成本约为

$$
O(N_p)+O(N_g\log N_g).
$$

代价是只能解析网格允许的空间尺度：PM 不能准确恢复远小于一个网格间距的两体引力。

下文使用标记 **粒子编号 $p$**、**网格索引 $\mathbf j$** 和 **Fourier 模式 $\mathbf n$**。

## 2. 目标方程

### 2.1 从共动坐标到无量纲变量

物理坐标与共动坐标满足 $\mathbf r=a\mathbf x$。扣除均匀背景的膨胀后，扰动势 $\Phi$ 满足

$$
\ddot{\mathbf x}+2H\dot{\mathbf x}
=-\frac{1}{a^2}\nabla_x\Phi,
\qquad
\nabla_x^2\Phi=4\pi G a^2\bar\rho_m(a)\delta.
$$

这里 $\delta=\rho_m/\bar\rho_m-1$ 是密度对比度，$\bar\rho_m$ 为物理平均密度。均匀背景已经由尺度因子 $a$ 描述，不再作为周期盒内的引力源。

非相对论物质有

$$
\bar\rho_m(a)=\frac{3H_0^2\Omega_m}{8\pi G}a^{-3},
\qquad
\nabla_x^2\Phi=\frac{3H_0^2\Omega_m}{2a}\delta.
$$

定义每单位质量的正则动量 $\mathbf p=a^2\dot{\mathbf x}$，则

$$
\dot{\mathbf p}
=2a\dot a\dot{\mathbf x}+a^2\ddot{\mathbf x}
=-\nabla_x\Phi.
$$

本项目以共动盒长 $L$ 和 $H_0$ 为尺度，定义

$$
\mathbf y=\frac{\mathbf x}{L},\qquad
\mathbf u=\frac{\mathbf p}{H_0L},\qquad
\psi=\frac{a\Phi}{H_0^2L^2},\qquad
\mathcal E(a)=\frac{H(a)}{H_0}.
$$

由于 $\nabla_x=L^{-1}\nabla_y$，得到

$$
\nabla_x^2\Phi=\frac{H_0^2}{a}\nabla_y^2\psi
\quad\Longrightarrow\quad
\boxed{\nabla_y^2\psi=C\delta,\qquad C\equiv\frac32\Omega_m.}
$$

利用 $da/dt=aH_0\mathcal E(a)$，运动方程成为

$$
\boxed{
\frac{d\mathbf y}{da}=\frac{\mathbf u}{a^3\mathcal E(a)},\qquad
\frac{d\mathbf u}{da}=\frac{\mathbf F}{a^2\mathcal E(a)},\qquad
\mathbf F=-\nabla_y\psi.
}
$$

粒子网格方法的目标就是求解无量纲力 $\mathbf F$，进而利用积分器计算 $\mathbf u$ 和 $\mathbf y$ 的演化。

### 2.2 输入、输出与网格约定

| 量 | 含义 | 程序中的约定 |
| --- | --- | --- |
| $\mathbf y_p$ | 粒子无量纲共动位置 | 三个分量都在 $[0,1)$ |
| $N_p$ | 粒子总数 | 正整数 |
| $N$ | 每方向网格点数 | 正整数 |
| $\Delta=1/N$ | 无量纲网格间距 | 物理共动间距为 $L/N$ |
| $\mathbf y_{\mathbf j}=\mathbf j/N$ | 网格点位置 | $j_x,j_y,j_z=0,\ldots,N-1$ |
| $\delta_{\mathbf j}$ | 网格密度对比度 | 无量纲 |
| $\mathbf F_{\mathbf j}$ | 网格力 | 三个实数数组 |
| $\mathbf F_p$ | 粒子力 | 与输入粒子一一对应 |

这里明确约定**网格点位于 $\mathbf j/N$**，密度和三个力分量都放在同一套点上。

周期边界使用

$$
y\leftarrow y-\lfloor y\rfloor.
$$

## 3. 用 CIC 将粒子分配到网格

### 3.1 从一维线性权重理解 CIC

一个粒子通常不在网格点上。将一维位置写成

$$
s=Ny,\qquad i=\lfloor s\rfloor,\qquad f=s-i\in[0,1).
$$

粒子位于网格点 $i$ 与 $i+1$ 之间，给两端的权重分别为

$$
w_0=1-f,\qquad w_1=f.
$$

这就是线性插值的权重：越靠近哪个点，分给哪个点的质量就越多。两端索引都按 $N$ 取模，因而最右端与最左端相连。

### 3.2 三维就是三个一维权重相乘

分别计算 $\mathbf i=\lfloor N\mathbf y_p\rfloor$ 和 $\mathbf f=N\mathbf y_p-\mathbf i$。对 $b_x,b_y,b_z\in\{0,1\}$，八个相邻点为

$$
\mathbf j(\mathbf b)=(\mathbf i+\mathbf b)\bmod N,
$$

对应权重为

$$
W_{p,\mathbf b}=w_{b_x}(f_x)w_{b_y}(f_y)w_{b_z}(f_z),
\qquad w_0(f)=1-f,\quad w_1(f)=f.
$$

由于

$$
\sum_{\mathbf b}W_{p,\mathbf b}
=[(1-f_x)+f_x][(1-f_y)+f_y][(1-f_z)+f_z]=1,
$$

CIC 分配下保持粒子总质量守恒。

### 3.3 从分配结果得到密度对比度

等质量粒子可先累加“有效粒子数”网格

$$
c_{\mathbf j}=\sum_p W_{\mathbf j}(\mathbf y_p),
\qquad \sum_{\mathbf j}c_{\mathbf j}=N_p.
$$

$W_{\mathbf j}$ 在该粒子的八个相邻点之外为零。若单粒子质量为 $m$，网格代表的共动体积为 $(L/N)^3$，则

$$
\rho_{\mathrm{com},\mathbf j}=\frac{mc_{\mathbf j}}{(L/N)^3},
\qquad
\bar\rho_{\mathrm{com}}=\frac{mN_p}{L^3}.
$$

物理密度都再除以 $a^3$，但在密度比值中抵消。因此

$$
\boxed{\delta_{\mathbf j}=\frac{N^3}{N_p}c_{\mathbf j}-1.}
$$

所以实现等质量 PM 无需先算出以太阳质量计的粒子质量。只有在 $N_p=N^3$ 时才能简化为 $\delta=c-1$。

> [!NOTE]
> 理论上 $\sum_{\mathbf j}\delta_{\mathbf j}=0$。实际应记录这项残差检查质量守恒；微小舍入残差可以由后面的零模处理消除，但不能用它掩盖错误的质量归一化。

若以后支持不等质量粒子，改为累加 $M_{\mathbf j}=\sum_p m_pW_{\mathbf j}(\mathbf y_p)$，再使用 $\delta_{\mathbf j}=N^3M_{\mathbf j}/M_{\rm total}-1$。

## 4. 使用傅里叶变换求解 Poisson 方程

### 4.1 周期盒中的 Fourier 展开

单位周期盒的基函数是 $e^{i\mathbf q\cdot\mathbf y}$，其中

$$
\mathbf q=2\pi\mathbf n,\qquad \mathbf n\in\mathbb Z^3.
$$

这里 $\mathbf q$ 是对无量纲坐标 $\mathbf y$ 的波矢；通常以 $h/\mathrm{Mpc}$ 等单位表示的物理共动波数为 $\mathbf k=\mathbf q/L$。本求解器使用 $q$。

Fourier 方法有用的原因是微分变成乘法：

$$
\nabla_y e^{i\mathbf q\cdot\mathbf y}=i\mathbf q e^{i\mathbf q\cdot\mathbf y},
\qquad
\nabla_y^2e^{i\mathbf q\cdot\mathbf y}=-q^2 e^{i\mathbf q\cdot\mathbf y}.
$$

因此每个非零模式分别满足

$$
-q^2\widetilde\psi=C\widetilde\delta,
\qquad
\widetilde\psi=-\frac{C}{q^2}\widetilde\delta.
$$

再使用 $\mathbf F=-\nabla_y\psi$，得到核心公式

$$
\boxed{\widetilde F_\alpha(\mathbf q)
=iC\frac{q_\alpha}{q^2}\widetilde\delta(\mathbf q),
\quad \alpha\in\{x,y,z\},\quad q^2>0.}
$$

### 4.2 零模式单独处理

对周期 Poisson 方程在整个盒子积分，周期边界上的通量相互抵消：

$$
\int\nabla_y^2\psi\,d^3y=0=C\int\delta\,d^3y.
$$

这要求源项平均值为零。密度零模代表均匀背景，势的任意常数也不影响力。因此规定

$$
\widetilde\psi(\mathbf0)=0,\qquad
\widetilde{\mathbf F}(\mathbf0)=\mathbf0.
$$

代码必须先判断 $q^2\neq0$，再做除法。

## 5. 离散傅里叶变换

### 5.1 正反变换归一化

采用负指数正变换：

$$
\widehat\delta_{\mathbf n}
=\sum_{j_x,j_y,j_z=0}^{N-1}
\delta_{\mathbf j}\exp\!\left(-\frac{2\pi i}{N}\mathbf n\cdot\mathbf j\right).
$$

相应的数学逆变换为

$$
\delta_{\mathbf j}=\frac1{N^3}\sum_{\mathbf n}
\widehat\delta_{\mathbf n}\exp\!\left(+\frac{2\pi i}{N}\mathbf n\cdot\mathbf j\right).
$$

对应 PyTorch 默认使用的 `norm="backward"` ，`torch.fft.rfftn` 保留原始 DFT 系数，`torch.fft.irfftn` 在逆变换中包含 $1/N^3$ 的归一化因子。 

[PyTorch `irfftn` 文档](https://docs.pytorch.org/docs/stable/generated/torch.fft.irfftn.html)

### 5.2 实空间布局与 R2C 半谱布局

使用 out-of-place 变换，且 $z$ 为连续存储方向。实数数组为 $N\times N\times N$：

```python
real_index = (ix * N + iy) * N + iz
```

R2C 输出是 $N\times N\times(\lfloor N/2\rfloor+1)$ 个复数：

```python
nz_complex = N // 2 + 1
complex_index = (ix * N + iy) * nz_complex + iz
```

这里 `ix, iy = 0 ... N-1`，但 `iz = 0 ... N/2`。

因为实数场有 Hermitian 对称

$$
\widehat\delta(-\mathbf n)=\widehat\delta(\mathbf n)^*,
$$

所以 `rfftn` 只存储最后一个方向的非负频率，共 $N^2(\lfloor N/2\rfloor + 1)$ 个复数。将这些系数直接传给 `irfftn`，它会利用对称性恢复实数场。（需要显式传入 `s=(N, N, N)`，使奇数网格也能恢复原始形状）

[PyTorch `rfftn` 文档](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfftn.html)

### 5.3 数组下标不等于有符号频率

对于前两个完整方向，定义

$$
n(i)=
\begin{cases}
i,&i\le\lfloor N/2\rfloor,\\
i-N,&i>\lfloor N/2\rfloor.
\end{cases}
$$

例如 $N=8$，数组索引 $0,1,2,3,4,5,6,7$ 对应 $0,1,2,3,4,-3,-2,-1$。最后一个半谱方向直接取 $n_z=i_z$。随后计算

$$
(q_x,q_y,q_z)=2\pi(n_x,n_y,n_z),\qquad
q^2=q_x^2+q_y^2+q_z^2.
$$

### 5.4 Nyquist 模式处理

为了保持力场的实数性，要求

$$
q_\alpha^{(D)}=
\begin{cases}
0,&N\text{ 为偶数且 }i_\alpha=N/2,\\
q_\alpha,&\text{其他情况}.
\end{cases}
$$

实际实现的力核为

$$
\boxed{\widehat F_\alpha
=iC\frac{q_\alpha^{(D)}}{q_x^2+q_y^2+q_z^2}\widehat\delta,
\qquad \mathbf q\ne0.}
$$

**分母保留原来的 $q^2$；只有当前求导方向的分子置零。** 例如 $n_x=N/2$、$n_y=1$、$n_z=0$ 时，$F_x$ 的该模式为零，但 $F_y$ 的该模式通常不为零。不要把所有含 Nyquist 分量的模式全部删除。奇数 $N$ 没有 Nyquist 格点，无需此特殊处理。

### 5.5 复数乘法可以直接展开

若 $\widehat\delta=A+iB$，令 $b_\alpha=Cq_\alpha^{(D)}/q^2$，则

$$
\widehat F_\alpha=ib_\alpha(A+iB)=-b_\alpha B+i b_\alpha A.
$$

对应代码就是

```python
force_hat_re = -b * delta_hat_im
force_hat_im = b * delta_hat_re
```

这也提供了检查引力符号的方法。

## 6. 第四步：把网格力插值回粒子

对三个力谱调用 `irfftn`，由逆变换完成归一化，得到 $\mathbf F_{\mathbf j}$。插值使用分配质量时的同一组八个 CIC 权重。

$$
\boxed{\mathbf F_p=\sum_{\mathbf b\in\{0,1\}^3}
W_{p,\mathbf b}\,\mathbf F_{\mathbf j(\mathbf b)}.}
$$

分配时是“一个粒子向八个网格点累加”；插值时是“一个粒子从八个网格点加权读取”。这里不再乘粒子质量、不再除以网格体积，也不再乘 $a$ 或 $H_0$。

相同的分配与插值配合反对称的力核，可以在精确算术下消除粒子自己的网格力，并使总内力为零。简要地，若分配矩阵为 $A$、某方向的网格力算子为 $K$，则插值是 $A^T$，而该谱力核给出 $K^T=-K$。于是有效粒子算子 $A^TKA$ 也是反对称的：对角元为零，粒子对的贡献相消。浮点实现允许有小的求和残差。

## 7. 与积分器耦合

为了理解模块连接，定义漂移与踢动因子

$$
\mathcal D(a_0,a_1)=\int_{a_0}^{a_1}\frac{da}{a^3\mathcal E(a)},\qquad
\mathcal K(a_0,a_1)=\int_{a_0}^{a_1}\frac{da}{a^2\mathcal E(a)}.
$$

对一步 $a_n\to a_{n+1}$，取 $a_m=(a_n+a_{n+1})/2$，使用 KDK 方法：

$$
\begin{aligned}
\mathbf F_n&=\operatorname{PM}(\mathbf y_n),\\
\mathbf u_{1/2}&=\mathbf u_n+\mathcal K(a_n,a_m)\mathbf F_n,\\
\mathbf y_{n+1}&=\operatorname{wrap}\!\left[
\mathbf y_n+\mathcal D(a_n,a_{n+1})\mathbf u_{1/2}\right],\\
\mathbf F_{n+1}&=\operatorname{PM}(\mathbf y_{n+1}),\\
\mathbf u_{n+1}&=\mathbf u_{1/2}+\mathcal K(a_m,a_{n+1})\mathbf F_{n+1}.
\end{aligned}
$$

初次计算以后可复用上一步的末力，因此正常每步只新增一次 PM 调用。力必须对应当前的位置，不能在漂移之后继续使用旧位置的力完成末次 kick。
