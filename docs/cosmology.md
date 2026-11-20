# 宇宙学背景、粒子运动与初始条件

本文系统推导项目所采用的平直 $\Lambda\mathrm{CDM}$ 宇宙学背景方程、共动坐标下的无量纲粒子运动方程、线性密度扰动增长理论，以及一阶拉格朗日微扰理论（1LPT）与玻尔兹曼求解器 CLASS 结合的初始条件构造方案。模拟粒子代表全部非相对论物质（冷暗物质与重子合并为单流体），背景膨胀遵循平直 Friedmann 方程，粒子扰动采用弱引力场、非相对论、远小于 Hubble 尺度下的牛顿引力近似，暂不考虑空间曲率、气体压力与重子反馈。

## 1. 尺度因子、红移与 Hubble 参数

### 1.1 共动坐标与红移

在均匀各向同性膨胀的宇宙中，粒子的物理坐标 $\mathbf r(t)$ 可分解为尺度因子 $a(t)$ 与共动坐标 $\mathbf x(t)$ 的乘积：

$$
\boxed{\mathbf r(t) = a(t)\mathbf x(t).}
$$

其中 $a(t)$ 为无量纲尺度因子，约定今天 $a_0 = 1$。若粒子仅随背景膨胀，其共动坐标 $\mathbf x$ 保持不变，物理距离随 $a(t)$ 成比例增长；因此，固定共动边长为 $L$ 的立方模拟盒，其物理边长为 $a(t)L$。

自由传播光子的波长随宇宙膨胀线性伸长，故相对今天的宇宙学红移 $z$ 满足：

$$
1 + z = \frac{1}{a}, \qquad a = \frac{1}{1+z}.
$$

Hubble 参数刻画宇宙的瞬时相对膨胀率：

$$
H(t) = \frac{\dot a}{a}, \qquad H_0 = H(a=1), \qquad \mathcal E(a) = \frac{H(a)}{H_0}.
$$

其中点号表示对宇宙时间 $t$ 求导，$H$ 具有时间倒数量纲，$\mathcal E(a)$ 为无量纲膨胀函数。通常将今天的 Hubble 常数参数化为

$$
H_0 = 100\,h\;\mathrm{km\cdot s^{-1}\cdot Mpc^{-1}},
$$

其中 $h$ 为无量纲 Hubble 参数（默认取 $h = 0.674$）。

### 1.2 物理速度与本动速度

对 $\mathbf r(t) = a(t)\mathbf x(t)$ 关于时间 $t$ 求导：

$$
\dot{\mathbf r} = \dot a\mathbf x + a\dot{\mathbf x}
= \frac{\dot a}{a}(a\mathbf x) + a\dot{\mathbf x}
= H\mathbf r + \mathbf v_{\rm pec}, \qquad
\boxed{\mathbf v_{\rm pec} = a\dot{\mathbf x}.}
$$

上式将物理速度分解为随背景膨胀的 Hubble 流 $H\mathbf r$ 与偏离均匀膨胀的本动速度 $\mathbf v_{\rm pec}$。宇宙学 $N$ 体模拟的核心是追踪粒子相对于均匀背景的本动演化。

## 2. 背景平均密度与 Friedmann 方程

### 2.1 流体连续性方程与状态方程

考虑物理体积 $V(t) = a(t)^3 V_0$ 的均匀共动区域，由热力学第一定律（绝热膨胀，无热量交换），区域内总能量 $U = \bar\rho c^2 V$ 的变化等于压强所做功的负值：

$$
d(\bar\rho c^2 V) = -P\,dV,
$$

其中 $\bar\rho$ 为平均质量等效密度，$P$ 为压强，$c$ 为光速。两边对时间 $t$ 求导并展开乘积导数：

$$
\dot{\bar\rho}c^2 V + \bar\rho c^2 \dot V = -P\dot V.
$$

两边同除以 $c^2 V$，并利用 $\dot V / V = 3\dot a / a = 3H$，得到宇宙学流体连续性方程：

$$
\dot{\bar\rho} + 3H\left(\bar\rho + \frac{P}{c^2}\right) = 0.
$$

若各组分互不交换能量，且满足常数状态方程 $P = w\bar\rho c^2$，则

$$
\dot{\bar\rho} = -3(1+w)\frac{\dot a}{a}\bar\rho
\;\Longrightarrow\;
\frac{d\ln\bar\rho}{d\ln a} = -3(1+w).
$$

从今天 $a=1$（密度 $\bar\rho_0$）积分至任意尺度因子 $a$，得：

$$
\ln\frac{\bar\rho(a)}{\bar\rho_0} = -3(1+w)\ln a
\;\Longrightarrow\;
\boxed{\bar\rho(a) = \bar\rho_0\,a^{-3(1+w)}.}
$$

平直 $\Lambda\mathrm{CDM}$ 背景包含以下三类组分：

| 组分 | 状态方程参数 $w$ | 平均密度演化规律 | 模型处理方式 |
| --- | ---: | --- | --- |
| 非相对论物质（CDM + 重子） | $0$ | $\bar\rho_m(a) = \bar\rho_{m,0}a^{-3}$ | 背景膨胀与粒子引力扰动均包含 |
| 辐射（光子与无质量中微子） | $1/3$ | $\bar\rho_r(a) = \bar\rho_{r,0}a^{-4}$ | 背景膨胀与初始传递函数包含；单流体线性增长基准中略去 |
| 宇宙学常数（暗能量） | $-1$ | $\rho_\Lambda = \text{const}$ | 仅进入均匀背景膨胀 |

### 2.2 临界密度与平直 Friedmann 方程

定义今天的临界密度与各组分的今日无量纲密度参数：

$$
\rho_{{\rm crit},0} = \frac{3H_0^2}{8\pi G}, \qquad
\Omega_m = \frac{\bar\rho_{m,0}}{\rho_{{\rm crit},0}}, \qquad
\Omega_r = \frac{\bar\rho_{r,0}}{\rho_{{\rm crit},0}}, \qquad
\Omega_\Lambda = \frac{\rho_\Lambda}{\rho_{{\rm crit},0}}.
$$

由此，任意时刻的非相对论物质平均物理密度为

$$
\boxed{\bar\rho_m(a) = \frac{3H_0^2\Omega_m}{8\pi G}\,a^{-3}.}
$$

本文约定：不带自变量的 $\Omega_m, \Omega_r, \Omega_\Lambda$ 始终表示今天的常数值。平直宇宙（$\Omega_k = 0$）的 Friedmann 方程为

$$
H^2(a) = \frac{8\pi G}{3}\big(\bar\rho_m(a) + \bar\rho_r(a) + \rho_\Lambda\big).
$$

将各组分密度代入，两边同除以 $H_0^2 = \frac{8\pi G}{3}\rho_{{\rm crit},0}$ 并取对应膨胀宇宙的正平方根，得到无量纲膨胀函数：

$$
\boxed{\mathcal E(a) = \sqrt{\Omega_r a^{-4} + \Omega_m a^{-3} + \Omega_\Lambda}, \qquad \Omega_m + \Omega_r + \Omega_\Lambda = 1.}
$$

显然 $\mathcal E(1) = 1$。在忽略辐射的晚期物质-暗能量宇宙中，上式简化为 $\mathcal E(a) = \sqrt{\Omega_m a^{-3} + \Omega_\Lambda}$。

### 2.3 瞬时物质占比与时间变量替换

任意时刻的临界密度为 $\rho_{\rm crit}(a) = \frac{3H(a)^2}{8\pi G} = \rho_{{\rm crit},0}\mathcal E(a)^2$，因此瞬时物质密度参数为

$$
\boxed{\Omega_m(a) = \frac{\bar\rho_m(a)}{\rho_{\rm crit}(a)} = \frac{\Omega_m a^{-3}}{\mathcal E(a)^2}.}
$$

对 $\mathcal E(a)^2 = \Omega_r a^{-4} + \Omega_m a^{-3} + \Omega_\Lambda$ 两边关于 $a$ 求导（下文撇号均表示 $d/da$）：

$$
2\mathcal E\mathcal E' = -4\Omega_r a^{-5} - 3\Omega_m a^{-4}
\;\Longrightarrow\;
\frac{\mathcal E'}{\mathcal E} = -\frac{2\Omega_r a^{-4} + \frac{3}{2}\Omega_m a^{-3}}{a\mathcal E(a)^2}.
$$

在忽略辐射项的近似下，上式化简为：

$$
\boxed{\frac{\mathcal E'}{\mathcal E} = -\frac{3}{2a}\Omega_m(a).}
$$

由于在膨胀宇宙中 $\dot a = aH(a) = aH_0\mathcal E(a) > 0$，$a(t)$ 与 $t$ 严格单调对应，可用尺度因子 $a$ 替代宇宙时间 $t$ 作为演化变量：

$$
\boxed{\frac{d}{dt} = \dot a\frac{d}{da} = aH_0\mathcal E(a)\frac{d}{da}, \qquad \frac{dt}{da} = \frac{1}{aH_0\mathcal E(a)}.}
$$

## 3. 扰动引力与共动运动方程

### 3.1 密度对比度与共动 Poisson 方程

将局部物质密度写作背景均值与无量纲密度对比度 $\delta(\mathbf r, t)$ 的形式：

$$
\rho_m(\mathbf r, t) = \bar\rho_m(t)\big[1 + \delta(\mathbf r, t)\big].
$$

均匀背景引起的全局膨胀加速度已完全包含在 $a(t)$ 中，粒子间的局部成团仅由超出背景的扰动引力势 $\Phi$ 驱动。在物理坐标下，$\Phi$ 满足牛顿 Poisson 方程：

$$
\nabla_r^2 \Phi = 4\pi G\big(\rho_m - \bar\rho_m\big) = 4\pi G\bar\rho_m(a)\delta.
$$

在固定时刻 $t$ 下，由 $\mathbf r = a\mathbf x$ 得梯度变换 $\nabla_r = a^{-1}\nabla_x$ 及拉普拉斯算子变换 $\nabla_r^2 = a^{-2}\nabla_x^2$。代入 $\bar\rho_m(a) = \frac{3H_0^2\Omega_m}{8\pi G}a^{-3}$，消去 $4\pi G$ 得共动坐标 Poisson 方程：

$$
a^{-2}\nabla_x^2\Phi = 4\pi G\cdot\frac{3H_0^2\Omega_m}{8\pi G}a^{-3}\delta
\;\Longrightarrow\;
\boxed{\nabla_x^2\Phi = \frac{3H_0^2\Omega_m}{2a}\delta.}
$$

宇宙学常数 $\Lambda$ 不产生空间扰动；同时周期性 Poisson 方程要求源项在全盒内的积分为零，这与 $\delta$ 零均值（$\langle\delta\rangle = 0$）的定义天然相容。

### 3.2 共动坐标运动方程与正则动量

在均匀背景中，随动粒子满足 $\mathbf r = a(t)\mathbf x$（$\mathbf x$ 为常数），其背景加速度为 $\ddot{\mathbf r}_{\rm bg} = \ddot a\,\mathbf x = \frac{\ddot a}{a}\mathbf r$。叠加扰动引力加速度 $-\nabla_r\Phi$ 后，粒子物理运动方程为：

$$
\ddot{\mathbf r} = \ddot a\,\mathbf x - \nabla_r\Phi = \ddot a\,\mathbf x - \frac{1}{a}\nabla_x\Phi.
$$

另一方面，对 $\mathbf r(t) = a(t)\mathbf x(t)$ 连续求两次时间导数：

$$
\dot{\mathbf r} = \dot a\mathbf x + a\dot{\mathbf x}, \qquad
\ddot{\mathbf r} = \ddot a\mathbf x + 2\dot a\dot{\mathbf x} + a\ddot{\mathbf x}.
$$

联立两式消去 $\ddot a\mathbf x$，得到：

$$
a\ddot{\mathbf x} + 2\dot a\dot{\mathbf x} = -\frac{1}{a}\nabla_x\Phi.
$$

两边同除以 $a$ 并利用 $H = \dot a/a$，得到共动运动方程：

$$
\boxed{\ddot{\mathbf x} + 2H\dot{\mathbf x} = -\frac{1}{a^2}\nabla_x\Phi.}
$$

其中 $2H\dot{\mathbf x}$ 为宇宙膨胀引起的哈勃阻尼项。为消去显式阻尼项并构造保辛（symplectic）积分格式，考虑单位质量粒子的共动拉格朗日量：

$$
\mathscr L(\mathbf x, \dot{\mathbf x}, t) = \frac{1}{2}a(t)^2|\dot{\mathbf x}|^2 - \Phi(\mathbf x, t).
$$

检验其 Euler–Lagrange 方程 $\frac{d}{dt}\left(\frac{\partial\mathscr L}{\partial\dot{\mathbf x}}\right) = \frac{\partial\mathscr L}{\partial\mathbf x}$：

$$
\frac{d}{dt}(a^2\dot{\mathbf x}) = -\nabla_x\Phi
\;\Longleftrightarrow\;
a^2\ddot{\mathbf x} + 2a\dot a\dot{\mathbf x} = -\nabla_x\Phi,
$$

与前述运动方程完全等价。由此自然定义每单位质量的**正则动量**：

$$
\mathbf p \equiv \frac{\partial\mathscr L}{\partial\dot{\mathbf x}} = a^2\dot{\mathbf x} = a\mathbf v_{\rm pec}.
$$

二阶运动方程遂化为不含显式摩擦项的一阶正则方程组：

$$
\boxed{\dot{\mathbf x} = \frac{\mathbf p}{a^2}, \qquad \dot{\mathbf p} = -\nabla_x\Phi.}
$$

由此可见，在无扰动引力时守恒的是正则动量 $\mathbf p$ 而非本动速度 $\mathbf v_{\rm pec} = \mathbf p/a$，后者随膨胀按 $a^{-1}$ 衰减。

## 4. 无量纲变量与方程体系

以共动盒长 $L$ 与今日 Hubble 常数 $H_0$ 为特征尺度，定义无量纲共动坐标 $\mathbf y \in [0,1)^3$、无量纲正则动量 $\mathbf u$ 与无量纲引力势 $\psi$：

$$
\mathbf y = \frac{\mathbf x}{L}, \qquad
\mathbf u = \frac{\mathbf p}{H_0 L}, \qquad
\psi = \frac{a\Phi}{H_0^2 L^2}.
$$

由 $\mathbf x = L\mathbf y$ 知 $\nabla_x = L^{-1}\nabla_y$、$\nabla_x^2 = L^{-2}\nabla_y^2$。将其与 $\Phi = \frac{H_0^2 L^2}{a}\psi$ 依次代入方程组：

1. **坐标演化方程**：利用 $\frac{d}{da} = \frac{1}{aH_0\mathcal E(a)}\frac{d}{dt}$ 与 $\dot{\mathbf x} = \mathbf p/a^2$，

$$
\frac{d\mathbf y}{da}
= \frac{1}{aH_0\mathcal E(a)}\cdot\frac{\dot{\mathbf x}}{L}
= \frac{1}{aH_0\mathcal E(a)}\cdot\frac{H_0 L\mathbf u}{a^2 L}
= \frac{\mathbf u}{a^3\mathcal E(a)}.
$$

2. **动量演化方程**：利用 $\dot{\mathbf p} = -\nabla_x\Phi = -\frac{1}{L}\nabla_y\left(\frac{H_0^2 L^2}{a}\psi\right) = -\frac{H_0^2 L}{a}\nabla_y\psi$，

$$
\frac{d\mathbf u}{da}
= \frac{1}{aH_0\mathcal E(a)}\cdot\frac{\dot{\mathbf p}}{H_0 L}
= -\frac{1}{aH_0\mathcal E(a)}\cdot\frac{H_0^2 L}{aH_0 L}\nabla_y\psi
= -\frac{\nabla_y\psi}{a^2\mathcal E(a)}.
$$

3. **无量纲 Poisson 方程**：将 $\Phi = \frac{H_0^2 L^2}{a}\psi$ 代入 $\nabla_x^2\Phi = \frac{3H_0^2\Omega_m}{2a}\delta$，

$$
\frac{1}{L^2}\cdot\frac{H_0^2 L^2}{a}\nabla_y^2\psi = \frac{3H_0^2\Omega_m}{2a}\delta
\;\Longrightarrow\;
\nabla_y^2\psi = \frac{3}{2}\Omega_m\delta.
$$

记无量纲引力场 $\mathbf F \equiv -\nabla_y\psi$，得到完整的无量纲动力学方程组：

$$
\boxed{
\frac{d\mathbf y}{da} = \frac{\mathbf u}{a^3\mathcal E(a)}, \qquad
\frac{d\mathbf u}{da} = \frac{\mathbf F}{a^2\mathcal E(a)}, \qquad
\mathbf F = -\nabla_y\psi, \qquad
\nabla_y^2\psi = \frac{3}{2}\Omega_m\delta.
}
$$

所有显式尺度因子依赖均已集中于时间系数 $a^{-3}\mathcal E^{-1}$ 与 $a^{-2}\mathcal E^{-1}$ 中，Poisson 方程系数恒为常数 $\frac{3}{2}\Omega_m$。无量纲量与物理量的对应关系如下：

| 无量纲变量 | 对应物理量 |
| --- | --- |
| 位置 $\mathbf y \in [0,1)^3$ | 共动位置 $\mathbf x = L\mathbf y$，物理位置 $\mathbf r = aL\mathbf y$ |
| 正则动量 $\mathbf u$ | $\mathbf p = H_0 L\mathbf u$，本动速度 $\mathbf v_{\rm pec} = H_0 L\mathbf u / a$ |
| 引力势 $\psi$ | 扰动势 $\Phi = H_0^2 L^2 \psi / a$ |
| 引力场 $\mathbf F$ | 物理扰动加速度 $-\nabla_r\Phi = H_0^2 L\mathbf F / a^2$ |

## 5. Kick 与 Drift 积分因子及数值求积

在算子分裂（Kick–Drift–Kick, KDK）格式中：
- **漂移（Drift）子步**固定动量 $\mathbf u$，对坐标方程在 $[a_0, a_1]$ 上积分：

$$
\Delta\mathbf y = \mathbf u\,\mathcal D(a_0, a_1), \qquad
\boxed{\mathcal D(a_0, a_1) = \int_{a_0}^{a_1} \frac{da}{a^3\mathcal E(a)}.}
$$

- **踢动（Kick）子步**固定坐标 $\mathbf y$（从而固定该位置所受引力 $\mathbf F$），对动量方程在 $[a_0, a_1]$ 上积分：

$$
\Delta\mathbf u = \mathbf F\,\mathcal K(a_0, a_1), \qquad
\boxed{\mathcal K(a_0, a_1) = \int_{a_0}^{a_1} \frac{da}{a^2\mathcal E(a)}.}
$$

对单步推进 $a_0 \to a_1$，取算术中点 $a_m = \frac{1}{2}(a_0 + a_1)$，KDK 一步更新为：

$$
\begin{aligned}
\mathbf u_{1/2} &= \mathbf u_0 + \mathcal K(a_0, a_m)\mathbf F(\mathbf y_0), \\
\mathbf y_1 &= \operatorname{wrap}\!\big[\mathbf y_0 + \mathcal D(a_0, a_1)\mathbf u_{1/2}\big], \\
\mathbf u_1 &= \mathbf u_{1/2} + \mathcal K(a_m, a_1)\mathbf F(\mathbf y_1).
\end{aligned}
$$

由于 $\mathcal E(a)$ 随 $a$ 变化，前后两个半步踢动因子 $\mathcal K(a_0, a_m)$ 与 $\mathcal K(a_m, a_1)$ 一般不相等，必须分别积分。

**对数尺度因子换元与 Gauss–Legendre 求积**：在宇宙学模拟中，跨多个数量级的尺度因子区间更适合在对数变量 $s = \ln a$ 下积分。利用 $a = e^s$ 与 $da = a\,ds$，将两个积分统一写作：

$$
I_p(a_0, a_1) = \int_{a_0}^{a_1} \frac{da}{a^{p+1}\mathcal E(a)}
= \int_{\ln a_0}^{\ln a_1} \frac{ds}{a(s)^p\,\mathcal E(a(s))},
$$

其中 $p=2$ 对应漂移因子 $\mathcal D$，$p=1$ 对应踢动因子 $\mathcal K$。再作仿射变换映射到标准区间 $\xi \in [-1, 1]$：

$$
s(\xi) = \frac{\ln a_0 + \ln a_1}{2} + \frac{\ln a_1 - \ln a_0}{2}\,\xi, \qquad
ds = \frac{\ln a_1 - \ln a_0}{2}\,d\xi.
$$

采用 8 节点 Gauss–Legendre 求积节点 $\{\xi_q\}_{q=1}^8$ 与权重 $\{w_q\}_{q=1}^8$，即得：

$$
\boxed{
I_p(a_0, a_1) \approx \frac{\ln a_1 - \ln a_0}{2}\sum_{q=1}^{8}
\frac{w_q}{a(\xi_q)^p\,\mathcal E(a(\xi_q))}.
}
$$

由于被积函数在 $\ln a$ 上高度光滑，8 节点高斯求积具有 15 次代数精度，其求积误差远小于粒子轨道离散化误差。

## 6. 线性密度扰动增长理论

在粒子轨道发生壳交叉（shell crossing）之前的早期线性阶段，可将非相对论物质近似为无压单流体（本节为推导简洁的理论基准，略去辐射项 $\Omega_r$）。线性增长因子 $D_+(a)$ 与对数增长率 $f(a)$ 用于阐明初始密度与动量的物理匹配关系，并作为解析检验的标尺。

### 6.1 流体线性化与时间增长方程

在固定共动坐标 $\mathbf x$ 下，无压流体的质量守恒（连续性方程）与动量守恒（Euler 方程）分别为：

$$
\frac{\partial\delta}{\partial t} + \frac{1}{a}\nabla_x\cdot\big[(1+\delta)\mathbf v\big] = 0,
$$

$$
\frac{\partial\mathbf v}{\partial t} + H\mathbf v + \frac{1}{a}(\mathbf v\cdot\nabla_x)\mathbf v = -\frac{1}{a}\nabla_x\Phi,
$$

其中 $\mathbf v = \mathbf v_{\rm pec}$，对流项 $\frac{1}{a}(\mathbf v\cdot\nabla_x)\mathbf v$ 源于沿粒子轨迹的物质导数 $\frac{d}{dt} = \frac{\partial}{\partial t} + \dot{\mathbf x}\cdot\nabla_x$。

当 $|\delta| \ll 1$ 且速度扰动为一阶小量时，略去二阶小量 $\delta\mathbf v$ 与 $(\mathbf v\cdot\nabla_x)\mathbf v$，并定义共动速度散度 $\theta \equiv \nabla_x\cdot\mathbf v$。对线性化 Euler 方程两边取共动散度，得到耦合的一阶线性方程组：

$$
\frac{\partial\delta}{\partial t} = -\frac{\theta}{a}, \qquad
\frac{\partial\theta}{\partial t} + H\theta = -\frac{1}{a}\nabla_x^2\Phi.
$$

对第一式两边关于 $t$ 求偏导，展开乘积导数：

$$
\frac{\partial^2\delta}{\partial t^2}
= -\frac{1}{a}\frac{\partial\theta}{\partial t} + \frac{\dot a}{a^2}\theta
= -\frac{1}{a}\frac{\partial\theta}{\partial t} + H\frac{\theta}{a}.
$$

将第二式改写为 $\frac{\partial\theta}{\partial t} = -H\theta - \frac{1}{a}\nabla_x^2\Phi$ 并代入上式：

$$
\frac{\partial^2\delta}{\partial t^2}
= -\frac{1}{a}\left(-H\theta - \frac{1}{a}\nabla_x^2\Phi\right) + H\frac{\theta}{a}
= 2H\frac{\theta}{a} + \frac{1}{a^2}\nabla_x^2\Phi.
$$

再利用第一式 $\frac{\theta}{a} = -\frac{\partial\delta}{\partial t}$ 与共动 Poisson 方程 $\nabla_x^2\Phi = 4\pi G a^2\bar\rho_m\delta$，整理即得线性密度增长方程：

$$
\boxed{\frac{\partial^2\delta}{\partial t^2} + 2H\frac{\partial\delta}{\partial t} - 4\pi G\bar\rho_m(a)\delta = 0.}
$$

该方程不含波数 $k$，因此在无压单流体近似下，所有尺度的线性密度模式共享同一时间演化函数。

### 6.2 转换为以尺度因子 $a$ 为自变量的方程

将线性增长模分离变量为 $\delta(\mathbf x, a) = D_+(a)\delta_0(\mathbf x)$，并约定今天归一化 $D_+(1) = 1$。利用 $\frac{d}{dt} = aH_0\mathcal E(a)\frac{d}{da}$，先计算一阶与二阶时间导数：

$$
\dot D_+ = aH_0\mathcal E\,D_+',
$$

$$
\begin{aligned}
\ddot D_+
&= aH_0\mathcal E\frac{d}{da}\big(aH_0\mathcal E\,D_+'\big) \\
&= aH_0^2\mathcal E\Big(\mathcal E\,D_+' + a\mathcal E'\,D_+' + a\mathcal E\,D_+''\Big) \\
&= a^2 H_0^2\mathcal E^2\left[D_+'' + \left(\frac{1}{a} + \frac{\mathcal E'}{\mathcal E}\right)D_+'\right].
\end{aligned}
$$

同时，哈勃阻尼项与引力源项分别为：

$$
2H\dot D_+ = 2(H_0\mathcal E)(aH_0\mathcal E\,D_+') = a^2 H_0^2\mathcal E^2\cdot\frac{2}{a}D_+',
$$

$$
4\pi G\bar\rho_m(a)D_+ = 4\pi G\cdot\frac{3H_0^2\Omega_m}{8\pi G}a^{-3}D_+ = \frac{3H_0^2\Omega_m}{2a^3}D_+.
$$

将三项代入增长方程，两边同除以 $a^2 H_0^2\mathcal E^2$，合并 $D_+'$ 系数（$\frac{1}{a} + \frac{2}{a} = \frac{3}{a}$），得到：

$$
\boxed{
D_+'' + \left(\frac{3}{a} + \frac{\mathcal E'}{\mathcal E}\right)D_+' - \frac{3\Omega_m}{2a^5\mathcal E^2}D_+ = 0.
}
$$

代入忽略辐射时的 $\frac{\mathcal E'}{\mathcal E} = -\frac{3}{2a}\Omega_m(a)$ 与 $\Omega_m(a) = \frac{\Omega_m a^{-3}}{\mathcal E^2}$，也可等价写作：

$$
D_+'' + \frac{3 - \frac{3}{2}\Omega_m(a)}{a}D_+' - \frac{3\Omega_m(a)}{2a^2}D_+ = 0.
$$

定义无量纲对数增长率：

$$
\boxed{f(a) = \frac{d\ln D_+}{d\ln a} = \frac{a D_+'(a)}{D_+(a)}.}
$$

## 7. 1LPT 初始条件与 CLASS 接入方案

### 7.1 一阶拉格朗日微扰理论（Zel'dovich 近似）

设粒子在初始均匀参考格点上的无量纲拉格朗日坐标为 $\mathbf Q \in [0,1)^3$，在初始尺度因子 $a_i$ 处的欧拉位置写作：

$$
\mathbf y(\mathbf Q, a_i) = \mathbf Q + \mathbf s_i(\mathbf Q),
$$

其中 $\mathbf s_i$ 为无量纲位移场。考虑参考格点处的微元体积 $d^3Q$，演化至位置 $\mathbf y$ 后变为共动体积 $d^3y = J\,d^3Q$，其中 $J$ 为坐标映射的雅可比行列式：

$$
J = \det\!\left(\frac{\partial y_\alpha}{\partial Q_\beta}\right)
= \det\!\left(\delta_{\alpha\beta} + \frac{\partial s_{i,\alpha}}{\partial Q_\beta}\right).
$$

由质量守恒 $\bar\rho_m(a_i)(a_i L)^3 d^3Q = \bar\rho_m(a_i)[1+\delta_i(\mathbf y)](a_i L)^3 d^3y$，消去公因子得 $1 + \delta_i = J^{-1}$。在小位移近似下，利用矩阵行列式一阶展开 $\det(I + \varepsilon) = 1 + \operatorname{tr}\varepsilon + O(\varepsilon^2)$ 与 $(1+x)^{-1} = 1 - x + O(x^2)$：

$$
1 + \delta_i = \frac{1}{1 + \nabla_Q\cdot\mathbf s_i + O(s_i^2)}
= 1 - \nabla_Q\cdot\mathbf s_i + O(s_i^2)
\;\Longrightarrow\;
\boxed{\nabla_Q\cdot\mathbf s_i = -\delta_i.}
$$

结合早期线性流动的无旋条件 $\nabla_Q\times\mathbf s_i = \mathbf 0$，在单位周期盒傅里叶空间（无量纲波矢 $\mathbf q = 2\pi\mathbf n$）中，有 $i\mathbf q\times\widetilde{\mathbf s}_i = \mathbf 0$ 与 $i\mathbf q\cdot\widetilde{\mathbf s}_i = -\widetilde\delta_i$。前者表明 $\widetilde{\mathbf s}_i$ 平行于 $\mathbf q$，设 $\widetilde{\mathbf s}_i = \alpha\mathbf q$ 并代入后者得 $i\alpha q^2 = -\widetilde\delta_i \Rightarrow \alpha = i\widetilde\delta_i / q^2$，从而唯一解出傅里叶空间位移场：

$$
\boxed{\widetilde{\mathbf s}_i(\mathbf q) = i\frac{\mathbf q}{q^2}\widetilde\delta_i(\mathbf q), \qquad \mathbf q \neq \mathbf 0.}
$$

### 7.2 单流体增长模的动量匹配关系

在单流体线性增长模型中，位移随时间满足 $\mathbf s(\mathbf Q, a) = D_+(a)\mathbf s_0(\mathbf Q)$。结合无量纲运动方程 $\frac{d\mathbf y}{da} = \frac{\mathbf u}{a^3\mathcal E(a)}$，初始正则动量必须满足：

$$
\mathbf u_i = a_i^3\mathcal E(a_i)\left.\frac{d\mathbf s}{da}\right|_{a_i}
= a_i^3\mathcal E(a_i)D_+'(a_i)\mathbf s_0
= a_i^2\mathcal E(a_i)\frac{a_i D_+'(a_i)}{D_+(a_i)}\mathbf s_i
= \boxed{a_i^2\mathcal E(a_i)f(a_i)\mathbf s_i.}
$$

若忽略初始动量（设 $\mathbf u_i = \mathbf 0$），将同时激发衰减模，导致早期线性增长偏离真实物理演化。

### 7.3 基于原初功率谱与 CLASS 的初始场生成方案

为精确包含重子与冷暗物质的尺度相关差异、辐射背景以及与牛顿 $N$ 体模拟自洽的规范定义，实际实验方案采用玻尔兹曼求解器 CLASS 在 $z_i = a_i^{-1} - 1$ 处提供的 N-body 规范传递函数，从同一份原初曲率扰动实现同步生成初始位移与动量：

1. **原初曲率功率谱与离散傅里叶振幅推导**：
   原初曲率扰动 $\mathcal R$ 采用幂律无量纲功率谱 $\Delta_{\mathcal R}^2(k) = A_s (k / k_{\rm pivot})^{n_s - 1}$（默认 $A_s = 2.1\times 10^{-9}$、$n_s = 0.965$、$k_{\rm pivot} = 0.05\,\mathrm{Mpc}^{-1}$），对应的三维功率谱为：

   $$
   P_{\mathcal R}(k) = \frac{2\pi^2}{k^3}\Delta_{\mathcal R}^2(k)
   = \frac{2\pi^2 A_s}{k_{\rm pivot}^3}\left(\frac{k}{k_{\rm pivot}}\right)^{n_s - 4},
   \qquad k = \frac{|\mathbf q|}{L}.
   $$

   在共动体积 $V = L^3$ 的周期盒中，连续场 $\mathcal R(\mathbf x)$ 展开为 $\mathcal R(\mathbf x) = \frac{1}{V}\sum_{\mathbf k}\tilde{\mathcal R}(\mathbf k)e^{i\mathbf k\cdot\mathbf x}$，其离散傅里叶模的方差满足 $\langle |\tilde{\mathcal R}(\mathbf k)|^2 \rangle = V P_{\mathcal R}(k)$。若在每轴 $N$ 个格点上采用非归一化正变换 $\widehat{\mathcal R}(\mathbf q) = \sum_{\mathbf j}\mathcal R(\mathbf x_{\mathbf j})e^{-i\mathbf q\cdot\mathbf y_{\mathbf j}}$，则 Riemann 求和给出 $\tilde{\mathcal R}(\mathbf k) \approx \frac{V}{N^3}\widehat{\mathcal R}(\mathbf q)$，因此：

   $$
   \langle |\widehat{\mathcal R}(\mathbf q)|^2 \rangle
   = \left(\frac{N^3}{V}\right)^2 V P_{\mathcal R}(k)
   = \frac{N^6}{L^3} P_{\mathcal R}(k).
   $$

   另一方面，对实空间独立同分布标准正态白噪声 $w_{\mathbf j} \sim \mathcal N(0, 1)$ 作同一非归一化正变换，得到 $W(\mathbf q) = \sum_{\mathbf j} w_{\mathbf j} e^{-i\mathbf q\cdot\mathbf y_{\mathbf j}}$，由独立性有 $\langle |W(\mathbf q)|^2 \rangle = \sum_{\mathbf j}\langle w_{\mathbf j}^2\rangle = N^3$。因此，构造原初曲率场的离散谱为：

   $$
   \boxed{\widehat{\mathcal R}(\mathbf q) = W(\mathbf q)\sqrt{\left(\frac{N}{L}\right)^3 P_{\mathcal R}(k)},}
   $$

   可严格检验 $\langle |\widehat{\mathcal R}(\mathbf q)|^2 \rangle = N^3 \cdot \frac{N^3}{L^3} P_{\mathcal R}(k) = \frac{N^6}{L^3} P_{\mathcal R}(k)$，且因白噪声为实数场，自动保证 Hermitian 共轭对称性。

2. **单流体传递函数合并**：
   从 CLASS 获取红移 $z_i$ 处 N-body 规范的重子与冷暗物质密度传递函数（$T_{\delta,b}, T_{\delta,{\rm cdm}}$）和速度散度传递函数（$T_{\theta,b}, T_{\theta,{\rm cdm}}$），按背景质量占比 $f_b = \frac{\omega_b}{\omega_b + \omega_{\rm cdm}}$ 加权合并为总物质单流体传递函数：

   $$
   T_\delta(k, z_i) = f_b T_{\delta,b} + (1 - f_b)T_{\delta,{\rm cdm}}, \qquad
   T_\theta(k, z_i) = f_b T_{\theta,b} + (1 - f_b)T_{\theta,{\rm cdm}}.
   $$

   在 $\ln k$ 空间对数插值到网格波数后，分别乘以同一份 $\widehat{\mathcal R}(\mathbf q)$ 得到相位一致的初始密度谱 $\widehat\delta_i(\mathbf q) = T_\delta(k, z_i)\widehat{\mathcal R}(\mathbf q)$ 与速度散度谱 $\widehat\theta_i(\mathbf q) = T_\theta(k, z_i)\widehat{\mathcal R}(\mathbf q)$。

3. **位移与无量纲正则动量的傅里叶反解**：
   CLASS 输出的速度散度定义为 $\theta_i = \nabla_x\cdot(\mathbf v_{\rm pec}/c)$（量纲为 $\mathrm{Mpc}^{-1}$）。将 $\mathbf v_{\rm pec} = \frac{H_0 L}{a_i}\mathbf u_i$ 与 $\nabla_x = L^{-1}\nabla_y$ 代入：

   $$
   \theta_i = \frac{1}{L}\nabla_y\cdot\left(\frac{H_0 L}{a_i c}\mathbf u_i\right)
   = \frac{H_0/c}{a_i}\,\nabla_y\cdot\mathbf u_i.
   $$

   在傅里叶空间中，$\nabla_y\cdot\mathbf u_i \leftrightarrow i\mathbf q\cdot\widehat{\mathbf u}_i$，结合无旋条件 $\mathbf q\times\widehat{\mathbf u}_i = \mathbf 0$，解得：

   $$
   i\mathbf q\cdot\widehat{\mathbf u}_i = \frac{a_i}{H_0/c}\widehat\theta_i
   \;\Longrightarrow\;
   \boxed{
   \widehat{\mathbf s}_i(\mathbf q) = i\frac{\mathbf q}{q^2}\widehat\delta_i(\mathbf q), \qquad
   \widehat{\mathbf u}_i(\mathbf q) = -i\frac{\mathbf q}{q^2}\frac{a_i}{H_0/c}\widehat\theta_i(\mathbf q).
   }
   $$

   将零模（$\mathbf q = \mathbf 0$）与所有 Nyquist 平面置零，经逆傅里叶变换回实空间得到格点位移 $\mathbf s_i$ 与初始动量 $\mathbf u_i$，并将粒子位置按周期边界折回 $\mathbf y_i = \operatorname{wrap}[\mathbf Q + \mathbf s_i]$。后续演化直接使用 CLASS 返回的自洽背景参数 $(\Omega_m, \Omega_\Lambda, \Omega_r)$ 计算 KDK 积分因子。

## 8. 粒子质量与 Einstein–de Sitter 解析检验方案

### 8.1 等质量粒子的质量与共动密度

固定共动盒的物理体积为 $(aL)^3$，总物质质量为 $M_{\rm box} = \bar\rho_m(a)(aL)^3 = \Omega_m\rho_{{\rm crit},0}L^3$，故 $N_p$ 个等质量粒子的单粒子质量为

$$
\boxed{m_p = \frac{\Omega_m\rho_{{\rm crit},0}L^3}{N_p}.}
$$

共动密度 $\rho_{\rm com} = a^3\rho_{\rm phys}$ 的空间均值不随时间变化，且密度对比度在物理密度与共动密度定义下完全相同：$\delta = \rho_{\rm com}/\bar\rho_{\rm com} - 1 = \rho_{\rm phys}/\bar\rho_{\rm phys} - 1$。因此 PM 求解器可直接由网格粒子计数构造 $\delta$。

### 8.2 Einstein–de Sitter（EdS）解析基准

令 $\Omega_m = 1, \Omega_\Lambda = 0, \Omega_r = 0$，即得 Einstein–de Sitter 宇宙，可用于严格检验数值积分器与初始动量标度：

1. **积分因子闭式解**：此时 $\mathcal E(a) = a^{-3/2}$，代入漂移与踢动积分定义直接手算得：

   $$
   \boxed{\mathcal D(a_0, a_1) = \int_{a_0}^{a_1} a^{-3/2}\,da = 2\big(a_0^{-1/2} - a_1^{-1/2}\big),}
   $$

   $$
   \boxed{\mathcal K(a_0, a_1) = \int_{a_0}^{a_1} a^{-1/2}\,da = 2\big(\sqrt{a_1} - \sqrt{a_0}\big).}
   $$

   在均匀网格粒子以常动量 $\mathbf u_0$ 平移的检验中（$\delta \equiv 0, \mathbf F \equiv \mathbf 0$），KDK 单步位移应精确等于 $\mathcal D(a_0, a_1)\mathbf u_0$。

2. **线性增长模解析解**：增长方程退化为 Euler–Cauchy 方程 $D'' + \frac{3}{2a}D' - \frac{3}{2a^2}D = 0$。代入试探解 $D = a^s$ 得特征方程：

   $$
   s(s-1) + \frac{3}{2}s - \frac{3}{2} = 0
   \;\Longleftrightarrow\;
   (s - 1)\left(s + \frac{3}{2}\right) = 0,
   $$

   故增长模与衰减模分别为 $D_+(a) = a$ 与 $D_-(a) \propto a^{-3/2}$，对应 $f(a) \equiv 1$。在早期高红移物质主导区间（如 $a_i = 0.05$），CLASS 生成的大尺度模式应近似满足 $u_i / (\delta_i a_i^2 \mathcal E(a_i)) \approx 1$，可作为初始条件动量归一化与符号的直接验证实验。
