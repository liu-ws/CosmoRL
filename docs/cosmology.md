# 宇宙学背景、粒子运动与线性增长

本文给出宇宙学公式的推导。模型采用平直 ΛCDM：非相对论物质和宇宙学常数，暂忽略辐射、空间曲率、气体压力与重子反馈。模拟粒子代表全部被建模的非相对论物质。

背景膨胀采用 Friedmann 方程；粒子扰动采用弱引力场、非相对论、远小于 Hubble 尺度下的 Newton 引力近似。

## 1. 尺度因子、红移与 Hubble 参数

### 1.1 共动坐标

宇宙膨胀，可将物理位置分解为

$$
\boxed{\mathbf r(t)=a(t)\mathbf x(t).}
$$

$\mathbf r$ 是物理坐标，$\mathbf x$ 是共动坐标，$a$ 是无量纲尺度因子，约定今天 $a_0=1$。若粒子只随背景膨胀，其 $\mathbf x$ 不变，物理距离仍随 $a$ 增长。因此，一个固定共动边长为 $L$ 的模拟盒，其物理边长是 $aL$。

光子波长随膨胀伸长，故相对今天的宇宙学红移满足

$$
1+z=\frac1a,\qquad a=\frac1{1+z}.
$$

Hubble 参数描述瞬时相对膨胀率：

$$
H(t)=\frac{\dot a}{a},\qquad H_0=H(a=1),\qquad
\mathcal E(a)=\frac{H(a)}{H_0}.
$$

点号表示对宇宙时间 $t$ 求导。$H$ 有时间倒数的量纲，$\mathcal E$ 无量纲。

$$
H_0=100h\;\mathrm{km \cdot s^{-1} \cdot Mpc^{-1}}.
$$

其中 $h\approx 0.7$ 为无量纲化的哈勃常数。

### 1.2 物理速度与本动速度

对 $\mathbf r=a\mathbf x$ 求导：

$$
\dot{\mathbf r}=\dot a\mathbf x+a\dot{\mathbf x}
=H\mathbf r+\mathbf v_{\rm pec},\qquad
\boxed{\mathbf v_{\rm pec}=a\dot{\mathbf x}.}
$$

$H\mathbf r$ 是背景 Hubble 流；$\mathbf v_{\rm pec}$ 是偏离均匀膨胀的本动速度，模拟主要追踪这种偏离。

## 2. 平均密度

### 2.1 连续性方程与状态方程

对体积 $V\propto a^3$ 的均匀共动区域，能量守恒写成

$$
d(\bar\rho c^2V)=-P\,dV.
$$

这里 $\bar\rho$ 是质量等效密度，$P$ 是压力，$c$ 是光速。展开并使用 $\dot V/V=3H$：

$$
\dot{\bar\rho}+3H\left(\bar\rho+\frac{P}{c^2}\right)=0.
$$

若组分之间不交换能量，且状态方程 $P=w\bar\rho c^2$ 中的 $w$ 为常数，则

$$
\frac{d\ln\bar\rho}{d\ln a}=-3(1+w),\qquad
\bar\rho(a)=\bar\rho_0a^{-3(1+w)}.
$$

| 组分 | $w$ | 密度随 $a$ 的变化 | 本项目 |
| --- | ---: | --- | --- |
| 非相对论物质 | 0 | $\bar\rho_m\propto a^{-3}$ | 包含 |
| 辐射 | $1/3$ | $\bar\rho_r\propto a^{-4}$ | 暂忽略 |
| 宇宙学常数 | $-1$ | $\rho_\Lambda$ 为常数 | 包含 |

### 2.2 临界密度与密度参数

今天的临界密度和物质密度参数定义为

$$
\rho_{\rm crit,0}=\frac{3H_0^2}{8\pi G},\qquad
\Omega_m=\frac{\bar\rho_{m,0}}{\rho_{\rm crit,0}},\qquad
\Omega_\Lambda=\frac{\rho_\Lambda}{\rho_{\rm crit,0}}.
$$

所以

$$
\boxed{\bar\rho_m(a)=\frac{3H_0^2\Omega_m}{8\pi G}a^{-3}.}
$$

本文不带自变量的 $\Omega_m$、$\Omega_\Lambda$ 始终表示**今天的参数**。它们是配置中的常数。

## 3. Friedmann 方程

平直宇宙的 Friedmann 方程为

$$
H^2=\frac{8\pi G}{3}(\bar\rho_m+\rho_\Lambda).
$$

代入上一节的密度，除以 $H_0^2$：

$$
\boxed{\mathcal E(a)=\sqrt{\Omega_ma^{-3}+\Omega_\Lambda},\qquad
\Omega_m+\Omega_\Lambda=1.}
$$

取正平方根对应膨胀的宇宙。对于本文的物质增长与粒子模型，要求 $a>0$、$\Omega_m>0$。平直条件应在配置阶段检查，$\mathcal E(1)=1$。

对照更一般的背景表达式，辐射和曲率会分别增加 $\Omega_ra^{-4}$ 与 $\Omega_ka^{-2}$ 项。

### 3.1 随时间变化的物质占比

任意时刻的临界密度为 $\rho_{\rm crit}(a)=3H(a)^2/(8\pi G)$，因此

$$
\boxed{\Omega_m(a)=\frac{\bar\rho_m(a)}{\rho_{\rm crit}(a)}
=\frac{\Omega_ma^{-3}}{\mathcal E(a)^2}.}
$$

在本模型的早期极限 $a\to0$，$\Omega_m(a)\to1$。

对膨胀函数求导可得

$$
2\mathcal E\mathcal E'=-3\Omega_ma^{-4},\qquad
\boxed{\frac{\mathcal E'}{\mathcal E}=-\frac{3}{2a}\Omega_m(a).}
$$

撇号在本文表示对 $a$ 求导。这个导数后面用于增长方程。

### 3.2 用尺度因子代替时间

由于 $\dot a=aH_0\mathcal E(a)>0$，可以用 $a$ 代替时间变量：

$$
\boxed{\frac{d}{dt}=aH_0\mathcal E(a)\frac{d}{da},\qquad
\frac{dt}{da}=\frac1{aH_0\mathcal E(a)}.}
$$

于是区间的物理时长为

$$
\Delta t=H_0^{-1}\int_{a_0}^{a_1}\frac{da}{a\mathcal E(a)}.
$$

若策略给出 $\Delta\ln a$，先转换为 $a_1=a_0\exp(\Delta\ln a)$，再按 $a$ 的方程积分。有限步长下不能直接把 $\Delta a$ 等同于 $\Delta\ln a$。

## 4. 扰动引力与共动运动方程

### 4.1 密度扰动

定义

$$
\rho_m(\mathbf r,t)=\bar\rho_m(t)[1+\delta(\mathbf r,t)].
$$

均匀背景造成的加速已由 $a(t)$ 描述；粒子求力只需要额外的扰动势 $\Phi$，其物理坐标 Poisson 方程为

$$
\nabla_r^2\Phi=4\pi G(\rho_m(a)-\bar\rho_m(a))=4\pi G\bar\rho_m(a)\delta.
$$

在固定时刻 $\nabla_r=a^{-1}\nabla_x$，并利用 $\bar\rho_m(a)=\frac{\bar\rho_{m,0}}{a^3}$

$$
\boxed{\nabla_x^2\Phi=4\pi Ga^2\bar\rho_m(a)\delta
=\frac{3H_0^2\Omega_m}{2a}\delta.}
$$

$\Lambda$ 在此模型中没有空间扰动，其作用进入背景膨胀，不作为额外的粒子密度网格。周期 Poisson 方程要求源项均值为零，这也解释了 PM 为什么使用 $\delta$ 而不是直接使用总密度。

### 4.2 运动方程

均匀背景提供加速度

$$
\mathbf r=a(t)\mathbf x
\quad\Rightarrow\quad
\ddot{\mathbf r}=\ddot a\,\mathbf x
=\frac{\ddot a}{a}\mathbf r.
$$

加入扰动后，均匀背景加速度仍取这个形式，方程变为

$$
\ddot{\mathbf r}=\frac{\ddot a}{a}\mathbf r-\nabla_r\Phi
=\ddot a\,\mathbf x-\nabla_r\Phi.
$$

另一方面，对 $\mathbf r=a\mathbf x$ 求两次时间导数：

$$
\ddot{\mathbf r}=\ddot a\mathbf x+2\dot a\dot{\mathbf x}+a\ddot{\mathbf x}.
$$

两式相等可得：

$$
2\dot a\dot{\mathbf x}+a\ddot{\mathbf x}
=-\frac1a\nabla_x\Phi.
$$

再除以 $a$：

$$
\boxed{\ddot{\mathbf x}+2H\dot{\mathbf x}=-\frac1{a^2}\nabla_x\Phi.}
$$

### 4.3 用正则动量吸收膨胀项

定义每单位质量的正则动量

$$
\mathbf p=a^2\dot{\mathbf x}=a\mathbf v_{\rm pec}.
$$

它可由单粒子比拉格朗日量 $\mathscr L=a^2|\dot{\mathbf x}|^2/2-\Phi$ 得到：$\partial\mathscr L/\partial\dot{\mathbf x}=\mathbf p$。直接求导也有

$$
\dot{\mathbf p}=a^2(\ddot{\mathbf x}+2H\dot{\mathbf x})=-\nabla_x\Phi.
$$

于是二阶方程化成

$$
\boxed{\dot{\mathbf x}=\frac{\mathbf p}{a^2},\qquad
\dot{\mathbf p}=-\nabla_x\Phi.}
$$

无扰动力时 $\mathbf p$ 不变，但 $\mathbf v_{\rm pec}=\mathbf p/a$ 随膨胀衰减。因此“无力”不等于“本动速度保持不变”。共动变量的常见约定也可参考 [Nyx 粒子运动方程](https://amrex-astro.github.io/Nyx/docs_html/Particles.html)，其速度记号与本文不同。

## 5. 本项目实际采取的无量纲变量

令共动盒长为 $L$，定义

$$
\mathbf y=\frac{\mathbf x}{L},\qquad
\mathbf u=\frac{\mathbf p}{H_0L},\qquad
\psi=\frac{a\Phi}{H_0^2L^2}.
$$

$\mathbf y$ 在单位周期盒 $[0,1)^3$ 内，$\mathbf u$ 是无量纲动量。依次使用 $d/da=(aH_0\mathcal E)^{-1}d/dt$：

$$
\frac{d\mathbf y}{da}
=\frac1L\frac{\mathbf p}{a^2}\frac1{aH_0\mathcal E}
=\frac{\mathbf u}{a^3\mathcal E},
$$

$$
\frac{d\mathbf u}{da}
=-\frac1{H_0L}\frac{H_0^2L}{a}\nabla_y\psi
\frac1{aH_0\mathcal E}
=-\frac{\nabla_y\psi}{a^2\mathcal E}.
$$

同理 Poisson 方程变成

$$
\frac{H_0^2}{a}\nabla_y^2\psi
=\frac{3H_0^2\Omega_m}{2a}\delta.
$$

最终供代码使用的方程是

$$
\boxed{
\frac{d\mathbf y}{da}=\frac{\mathbf u}{a^3\mathcal E(a)},\qquad
\frac{d\mathbf u}{da}=\frac{\mathbf F}{a^2\mathcal E(a)},\qquad
\mathbf F=-\nabla_y\psi,\qquad
\nabla_y^2\psi=\frac32\Omega_m\delta.
}
$$

这里 Poisson 系数用的是常数 $\Omega_m$，不是 $\Omega_m(a)$。所有显式的 $a$ 因子已移入变量定义与时间系数。

| 程序变量 | 转换回物理量 |
| --- | --- |
| 位置 $\mathbf y$ | 共动位置 $\mathbf x=L\mathbf y$，物理位置 $\mathbf r=aL\mathbf y$ |
| 动量 $\mathbf u$ | $\mathbf p=H_0L\mathbf u$，本动速度 $\mathbf v_{\rm pec}=H_0L\mathbf u/a$ |
| 势 $\psi$ | $\Phi=H_0^2L^2\psi/a$ |
| PM 力 $\mathbf F$ | 扰动物理加速度 $-\nabla_r\Phi=H_0^2L\mathbf F/a^2$ |

若盒长数值为 $L_{\rm box}$，单位是 $\mathrm{Mpc}/h$，则实际 $L=(L_{\rm box}/h)\,\mathrm{Mpc}$，从而 $H_0L=100L_{\rm box}\,\mathrm{km/s}$。转换时不能再重复乘一次 $h$。

## 6. Kick 与 Drift 积分因子

在只更新位置的 drift 子步骤中，固定 $\mathbf u$；在只更新动量的 kick 子步骤中，固定位置，因而固定该位置对应的 PM 力。分别积分上一节的方程：

$$
\Delta\mathbf y=\mathbf u\,\mathcal D(a_0,a_1),\qquad
\boxed{\mathcal D(a_0,a_1)=\int_{a_0}^{a_1}\frac{da}{a^3\mathcal E(a)}},
$$

$$
\Delta\mathbf u=\mathbf F\,\mathcal K(a_0,a_1),\qquad
\boxed{\mathcal K(a_0,a_1)=\int_{a_0}^{a_1}\frac{da}{a^2\mathcal E(a)}}.
$$

本文用花体 $\mathcal D$ 表示漂移积分，对应 PM 文档中的双参数 $\mathcal D(a_0,a_1)$；下一节的 $D_+(a)$ 表示线性增长因子，两者不是同一个函数。

取 $a_m=(a_0+a_1)/2$，KDK 一步为

$$
\begin{aligned}
\mathbf u_{1/2}&=\mathbf u_0+\mathcal K(a_0,a_m)\mathbf F(\mathbf y_0),\\
\mathbf y_1&=\operatorname{wrap}[\mathbf y_0+\mathcal D(a_0,a_1)\mathbf u_{1/2}],\\
\mathbf u_1&=\mathbf u_{1/2}+\mathcal K(a_m,a_1)\mathbf F(\mathbf y_1).
\end{aligned}
$$

两个 kick 因子一般不相等，也不应直接用整段积分的一半代替。积分因子精确计算的是各个拆分子步骤的背景系数，并不使整个粒子轨道成为精确解，因为真实轨道上的位置和力都在同时变化。

可使用 FP64 自适应数值求积计算这两个一维积分。其误差应小于准备测量的轨道积分误差；函数可按背景参数复用或缓存。

## 7. 密度扰动增长

增长因子用来生成初始条件、确定线性功率谱的时间缩放，以及验证低振幅 PM 演化。这里在壳交叉之前将冷物质视为无压单流体，仅在线性推导中使用这种描述。

### 7.1 连续性方程与 Euler 方程

用 $\mathbf v=\mathbf v_{\rm pec}$、共动导数 $\nabla_x$，密度对比度的连续性方程为

$$
\frac{\partial\delta}{\partial t}+\frac1a\nabla_x\cdot[(1+\delta)\mathbf v]=0.
$$

它可由物理坐标的质量守恒式，代入 $\rho=\bar\rho(1+\delta)$、$\frac{d\bar\rho}{dt}=-3H\bar\rho$ 并切换到固定共动坐标得到。

由运动方程可得本动速度满足 Euler 方程：

$$
\frac{\partial\mathbf v}{\partial t}+H\mathbf v
+\frac1a(\mathbf v\cdot\nabla_x)\mathbf v
=-\frac1a\nabla_x\Phi.
$$

这里的时间偏导均在固定共动位置 $\mathbf x$ 处计算；背景密度 $\bar\rho$ 仅依赖时间，使用普通导数。沿粒子轨迹的导数为 $d\mathbf v/dt=\frac{\partial\mathbf v}{\partial t}+(\mathbf v\cdot\nabla_x)\mathbf v/a$，因此将粒子运动方程写成速度场的方程时会出现对流项。

忽略二阶小量 $\delta\mathbf v$ 和 $(\mathbf v\cdot\nabla_x)\mathbf v$，并对线性化后的 Euler 方程取散度。定义 $\theta=\nabla_x\cdot\mathbf v$，得到

$$
\frac{\partial\delta}{\partial t}=-\frac\theta a,\qquad
\frac{\partial\theta}{\partial t}+H\theta=-\frac1a\nabla_x^2\Phi.
$$

对第一式求时间导数，并用第二式消去 $\frac{\partial\theta}{\partial t}$：

$$
\frac{\partial^2\delta}{\partial t^2}=-\frac1a\frac{\partial\theta}{\partial t}+\frac{H\theta}{a}
=\frac{2H\theta}{a}+\frac{\nabla_x^2\Phi}{a^2}
=-2H\frac{\partial\delta}{\partial t}+4\pi G\bar\rho_m\delta.
$$

所以

$$
\boxed{\frac{\partial^2\delta}{\partial t^2}+2H\frac{\partial\delta}{\partial t}-4\pi G\bar\rho_m\delta=0.}
$$

引力项促使过密区增长，膨胀项抑制增长。在本模型下方程没有波数依赖，因此所有线性模式共享同一个增长函数。不同尺度的初始振幅仍由输入功率谱决定。

### 7.2 改写成以 $a$ 为自变量的增长方程

选取增长模，写作 $\delta(\mathbf x,a)=D_+(a)\delta_0(\mathbf x)$，约定 $D_+(1)=1$。设 $A(a)=aH(a)$，链式法则给出

$$
\dot D_+=AD_+',\qquad
\ddot D_+=A^2D_+''+AA'D_+'.
$$

代入增长方程并除以 $A^2$：

$$
\boxed{
D_+''+\left[\frac3a+\frac{\mathcal E'}{\mathcal E}\right]D_+'
-\frac{3\Omega_m}{2a^5\mathcal E^2}D_+=0.
}
$$

等价地，使用瞬时物质占比：

$$
D_+''+\frac{3-\frac32\Omega_m(a)}aD_+'
-\frac{3\Omega_m(a)}{2a^2}D_+=0.
$$

转化为一阶方程，令 $V=D_+'$：

$$
\frac{dD_+}{da}=V,\qquad
\frac{dV}{da}=-\left(\frac3a+\frac{\mathcal E'}{\mathcal E}\right)V
+\frac{3\Omega_m}{2a^5\mathcal E^2}D_+.
$$

**这个只有两个标量的背景 ODE 很适合用 RK 方法求解**，它不包含昂贵的 PM 调用。可先求出增长函数表，再供粒子初始化和诊断使用。

### 7.3 增长模初值与归一化

选择模型内足够早、满足 $\Omega_\Lambda a_{\rm start}^3/\Omega_m\ll1$ 的起点，利用物质主导近似，把 $\Omega_m(a)\simeq1$ 代入 7.2 的方程：

$$
D_+''+\frac{3}{2a}D_+'-\frac{3}{2a^2}D_+=0.
$$

通解是

$$
D(a)=C_1a+C_2a^{-3/2}.
$$

选取增长模

$$
D_+=C_1 a
$$

先选择系数 $C_1=1$，给出积分初值

$$
D_{\rm raw}(a_{\rm start})=a_{\rm start},\qquad
D_{\rm raw}'(a_{\rm start})=1.
$$

积分到 $a=1$ 后，统一归一化，矫正系数：

$$
D_+(a)=\frac{D_{\rm raw}(a)}{D_{\rm raw}(1)},\qquad
D_+'(a)=\frac{D_{\rm raw}'(a)}{D_{\rm raw}(1)}.
$$

导数必须除以同一个常数，否则初始速度会错。为了检验忽略暗能量项的误差是否可以接受，可以用更早的起点复算，检查增长函数是否收敛。这里把起点前移需要在所选“物质加 Λ”近似模型内改善初值，不能早到早期辐射不可忽略的时期。

常用的对数增长率定义为

$$
\boxed{f(a)=\frac{d\ln D_+}{d\ln a}=\frac{aD_+'(a)}{D_+(a)}.}
$$

可由已求出的 $D_+$ 与导数直接计算，关于增长率及其与背景模型的关系，可参阅 [Polarski & Gannouji 的研究](https://arxiv.org/abs/0710.1510)。

## 8. 从增长因子得到 1LPT 初始条件

### 8.1 位移与密度的关系

用 $\mathbf Q$ 标记均匀分布的初始粒子位置，均在无量纲单位盒内。Zel'dovich（1LPT）近似写成

$$
\mathbf y(\mathbf Q,a)=\mathbf Q+D_+(a)\mathbf s(\mathbf Q).
$$

其中 $\mathbf s$ 是与时间无关的位移场。在均匀参考坐标中，取一小块体积 $d^3Q$。粒子移动后，它变成实际共动坐标中的体积 $d^3y$。参考分布的密度为平均密度 $\bar\rho_m(a)$，扰动后的密度为

$$
\rho(\mathbf y,a)=\bar\rho_m(a)[1+\delta(\mathbf y,a)].
$$

同一小团物质的质量不变，因此

$$
\bar\rho_m(a)(aL)^3d^3Q=
\bar\rho_m(a)[1+\delta(\mathbf y,a)](aL)^3d^3y.
$$

消去共同因子

$$
[1+\delta(\mathbf y,a)]\,d^3y=d^3Q,
\qquad
1+\delta=\frac1J,
$$

利用上面的近似位移公式，Jacobian 为

$$
J=\det\left[\delta_{ij}+D_+\frac{\partial s_i}{\partial Q_j}\right]
$$

对于小矩阵扰动 $\varepsilon$，行列式满足

$$
\det(I+\varepsilon)=1+\operatorname{tr}\varepsilon+O(\varepsilon^2).
$$

因此雅可比可表示为

$$
J=1+D_+\nabla_Q\cdot\mathbf s+O((D_+\nabla_Q\mathbf s)^2).
$$

保留一阶项：

$$
\delta=-D_+\nabla_Q\cdot\mathbf s.
$$

若 $\delta_0$ 表示线性外推到 $a=1$ 的密度扰动，则

$$
\boxed{\nabla_Q\cdot\mathbf s=-\delta_0.}
$$

这里 $\delta_0$ 是线性外推场，不是已经完成非线性演化的今天密度场。如果输入扰动位于 $a_i$，令 $\mathbf s_i=D_+(a_i)\mathbf s$，则有

$$
\nabla_Q\cdot\mathbf s_i=-\delta_i,\qquad \mathbf y=\mathbf Q+\mathbf s_i.
$$

散度方程还不能唯一确定 $\mathbf s$，进一步加上无旋条件，采用傅里叶变换

$$
\mathbf s(\mathbf Q)=\sum_{\mathbf q}\widetilde{\mathbf s}(\mathbf q)
e^{i\mathbf q\cdot\mathbf Q},
$$

其中 $\mathbf q=2\pi\mathbf n$ 为单位盒波矢，取旋度可得

$$
\nabla_Q\times\mathbf s=0
\quad\Longrightarrow\quad
\mathbf q\times\widetilde{\mathbf s}(\mathbf q)=0.
$$

每个傅里叶模式的位移振幅都沿着该模式的波矢方向，可以写成

$$
\widetilde{\mathbf s}(\mathbf q)=A(\mathbf q)\mathbf q.
$$

代入散度条件

$$
i\mathbf q\cdot\widetilde{\mathbf s}
=iAq^2=-\widetilde\delta_0,
$$

因为 $i\mathbf q\cdot(i\mathbf q\widetilde\delta_0/q^2)=-\widetilde\delta_0$，所以解得：

$$
\boxed{\widetilde{\mathbf s}(\mathbf q)
=i\frac{\mathbf q}{q^2}\widetilde\delta_0(\mathbf q).}
$$

### 8.2 初始动量不能随便设成零

由 $d\mathbf y/da=D_+'\mathbf s$ 和 $d\mathbf y/da=\mathbf u/(a^3\mathcal E)$：

$$
\boxed{
\mathbf y_i=\operatorname{wrap}[\mathbf Q+D_+(a_i)\mathbf s],\qquad
\mathbf u_i=a_i^3\mathcal E(a_i)D_+'(a_i)\mathbf s.
}
$$

也可以写成 
$$
\mathbf u_i=a_i^2\mathcal E(a_i)f(a_i)D_+(a_i)\mathbf s=a_i^2\mathcal E(a_i)f(a_i)\mathbf s_i
$$

若给有密度扰动的粒子任意设置零动量，通常会混入衰减模，无法得到期望的纯增长初始条件。

在线性理论中 $\delta\propto D_+$，功率谱是振幅的二次量，因此

$$
P_{\rm lin}(k,a)=D_+^2(a)P_{\rm lin}(k,1).
$$

### 8.3 实现中从 CLASS 读取初值

项目在 $z_i=a_i^{-1}-1$ 调用 CLASS，直接读取 N-body gauge 的重子和冷暗物质的密度与速度散度传递函数。各分量按背景质量占比合并为单一无碰撞流体，密度和速度共用一次高斯曲率场抽样。取盒长 $L$ 的单位为 Mpc，向 CLASS 查询的波数为 $k=q/L$。

令 $\theta_i=\nabla_{\mathbf x}\cdot(\mathbf v_{\rm pec}/c)$ 为 CLASS 速度散度，单位为 $\mathrm{Mpc}^{-1}$。用 CLASS 的输出在傅里叶空间分别构造位移与无量纲正则动量。

$$
\widetilde{\mathbf s}_i=i\frac{\mathbf q}{q^2}\widetilde\delta_i,\qquad
\widetilde{\mathbf u}_i=-i\frac{\mathbf q}{q^2}\frac{a_i}{H_0/c}\widetilde\theta_i.
$$

零模和初始粒子格上的 Nyquist 平面置零。后续以 Newtonian PM 演化粒子，时间积分因子使用 CLASS 给出的 $\Omega_m$、$\Omega_\Lambda$ 与 $\Omega_r$，因此膨胀率还包含 $\Omega_r a^{-4}$。线性增长方程保留作推导和解析对照。 [CLASS 输出参数](https://github.com/lesgourg/class_public/blob/master/explanatory.ini)

## 9. 粒子质量与模拟盒的总质量

固定共动盒中的物理体积是 $(aL)^3$，其平均物质质量为

$$
M_{\rm box}=\bar\rho_m(a)(aL)^3
=\Omega_m\rho_{\rm crit,0}L^3.
$$

所以等质量粒子的质量为

$$
\boxed{m_p=\frac{\Omega_m\rho_{\rm crit,0}L^3}{N_p}.}
$$

共动密度 $\rho_{\rm com}=a^3\rho_{\rm phys}$，其平均值恒定；但密度对比度在两种定义下相同：

$$
\frac{\rho_{\rm com}}{\bar\rho_{\rm com}}-1
=\frac{\rho_{\rm phys}}{\bar\rho_{\rm phys}}-1=\delta.
$$

这就是 PM 可以直接用网格粒子计数构造 $\delta$ 的原因。物理密度、共动密度与本动速度的区分也可参考 [Nyx 共动方程说明](https://amrex-astro.github.io/Nyx/docs_html/NyxEquations.html)。

## 10. 用 Einstein–de Sitter 宇宙做解析检查

令 $\Omega_m=1$、$\Omega_\Lambda=0$，称为 Einstein–de Sitter（EdS）宇宙。它是本项目非常有用的测试模型。

### 10.1 背景与积分因子

此时 $\mathcal E=a^{-3/2}$，所以

$$
\frac{dt}{da}=\frac{a^{1/2}}{H_0},\qquad
t(a)=\frac{2}{3H_0}a^{3/2},
$$

其中取大爆炸为 $t=0$。漂移、踢动积分可手算：

$$
\boxed{\mathcal D(a_0,a_1)=\int_{a_0}^{a_1}a^{-3/2}da
=2(a_0^{-1/2}-a_1^{-1/2}),}
$$

$$
\boxed{\mathcal K(a_0,a_1)=\int_{a_0}^{a_1}a^{-1/2}da
=2(\sqrt{a_1}-\sqrt{a_0}).}
$$

对于 $a_1>a_0>0$，二者均为正。可直接用这些解析表达式验证通用数值求积。

### 10.2 增长模与衰减模

增长方程简化为

$$
D''+\frac{3}{2a}D'-\frac{3}{2a^2}D=0.
$$

尝试幂律 $D=a^s$，得到

$$
s(s-1)+\frac32s-\frac32=0
\quad\Longrightarrow\quad (s-1)(s+3/2)=0.
$$

所以两个独立解为

$$
\boxed{D_+(a)=a,\qquad D_-(a)\propto a^{-3/2}.}
$$

增长模归一化后满足 $D_+(1)=1$、$D_+'=1$、$f=1$。一般 ΛCDM 的晚期增长不满足 $D_+=a$，只能在物质主导极限使用这个近似。

在 EdS 的 1LPT 初始条件中，还应得到 $\mathbf u_i=a_i^{3/2}\mathbf s$，可用于同时检查动量和增长导数。
