# 自适应步长强化学习、$O(3)$ 等变网络与 PPO 推导

本文阐述将宇宙学 Particle-Mesh（PM）模拟的自适应时间步长选择建模为有限预算马尔可夫决策过程（MDP）的理论框架，重点给出三维周期 $O(3)$ 等变卷积网络的设计思路与数学推导，并说明带动作掩码的 PPO 算法、GAE 优势估计与实验评估方案。

## 1. 自适应步长 MDP 建模

### 1.1 Episode 与状态观测

单个 episode 对应一次从初始尺度因子 $a_i$（默认 $0.05$）到终止尺度因子 $a_f$（默认 $1.0$）的完整 PM 宇宙学演化。每个决策步 $n$ 选择下一段 KDK 积分步长，训练与评估均在不同随机种子生成的初始场实现上进行，以检验策略对未见宇宙学实现的泛化能力。

完整的物理环境状态由粒子位置与正则动量、当前尺度因子 $a_n$ 及剩余步数预算 $B_{\rm rem}$ 决定。为构造固定维度且保留三维空间几何结构的观测 $o_n$，使用与 PM 求解器相同的 CIC 核将粒子质量与正则动量分配到 $N^3$ 周期网格上，得到：
- **标量场**：无量纲密度对比度场 $\delta(\mathbf y_{\mathbf j})$（1 通道）；
- **极矢量场**：网格化无量纲动量密度场 $\mathbf m(\mathbf y_{\mathbf j}) = \frac{N^3}{N_p}\sum_p W_{\mathbf j}(\mathbf y_p)\mathbf u_p$（3 通道，对应 $x, y, z$ 分量）；
- **全局标量上下文**：当前尺度因子与剩余预算占比 $\mathbf g_n = \big(a_n,\, B_{\rm rem}/B\big) \in \mathbb R^2$。

### 1.2 离散动作空间与预算掩码

策略在每步从离散的对数尺度因子步长候选集 $\mathcal H = \{h^{(1)}, \dots, h^{(M)}\}$ 中选择一个动作（默认取 $\mathcal H = \{0.01, 0.02, 0.04, 0.08\}$）。若当前尺度因子为 $a_n$，选择步长 $h$ 对应的目标尺度因子经终点截断为：

$$
a_{n+1}(h) = \min\!\big(a_n e^{h},\; a_f\big).
$$

定义计算预算 $B$ 为**初始化 PM 求解之后允许新增的 PM 力计算次数**（因 KDK 的 FSAL 特性，恰好等于允许的决策步数）。仅检查 $B_{\rm rem} \ge 1$ 不足以避免策略在前期耗尽预算而无法抵达 $a_f$。记候选集中最大步长为 $h_{\max} = \max\mathcal H$，从任意尺度因子 $a$ 抵达 $a_f$ 所需的最少步数为：

$$
C_{\min}(a) = \left\lceil \frac{\ln(a_f / a)}{h_{\max}} \right\rceil.
$$

一个候选动作 $h \in \mathcal H$ 合法，当且仅当执行该步后剩余预算仍足以在最大步长下到达终点，且不与此前更小步长因终点截断而产生重复的目标 $a_{n+1}$：

$$
\boxed{1 + C_{\min}\!\big(a_{n+1}(h)\big) \le B_{\rm rem}.}
$$

将每个动作的合法性记为二值掩码 $m_n(j) \in \{0, 1\}$。在策略网络输出原始对数几率（logits）$z_n \in \mathbb R^M$ 后，带掩码的动作概率分布定义为：

$$
\pi_\theta(j \mid o_n) = \frac{m_n(j)\exp(z_{n,j})}{\sum_{k=1}^M m_n(k)\exp(z_{n,k})},
$$

等价于将所有非法动作的 logit 置为 $-\infty$ 后做 Softmax。由此非法动作的采样概率严格为 $0$，且在策略熵 $H(\pi_\theta(\cdot\mid o_n)) = -\sum_{j: m_n(j)=1}\pi_\theta(j\mid o_n)\ln\pi_\theta(j\mid o_n)$ 与 PPO 更新的新旧对数概率计算中均复用 rollout 阶段保存的原始掩码。数值溢出（非有限值）或无合法动作均立即判定为失败终止。

## 2. 三维周期 $O(3)$ 等变卷积网络推导

### 2.1 不同阶特征的旋转响应与普通卷积的局限

三维周期宇宙学模拟盒具有显著的物理对称性：对整个盒子施加任意三维周期平移，或施加任意旋转与镜像反射 $R \in O(3)$（满足正交条件 $R^\top R = I$ 与 $\det R = \pm 1$），同一宇宙学实现的最佳积分步长与状态价值必须保持不变。

然而，模拟盒上的不同物理量对旋转与镜像反射具有不同的变换响应：
- **0 阶标量场 $s(\mathbf r)$**（如密度对比度 $\delta$）：在旋转或镜像反射下数值不变，满足 $s'(R\mathbf r) = s(\mathbf r)$；
- **1 阶极矢量场 $\mathbf v(\mathbf r)$**（如动量场 $\mathbf m$、网格相对方向 $\mathbf n$）：随坐标系同步旋转与反射，满足 $\mathbf v'(R\mathbf r) = R\mathbf v(\mathbf r)$；
- **1 阶轴矢量场 $\mathbf a(\mathbf r)$**（如两个极矢量叉积产生的涡度/角动量类特征）：旋转时与极矢量一致，但镜像反射时多出一个符号反转，满足 $\mathbf a'(R\mathbf r) = (\det R)\,R\mathbf a(\mathbf r)$。

普通三维卷积网络不满足 $O(3)$ 等变性，根本原因在于它将不同阶（以及不同镜像宇称）的特征当作普通数值通道直接线性混合，并对矢量分量逐个施加非线性激活。一旦坐标系发生旋转，标量分量不变而矢量各分量会按 $R$ 重新线性组合，原本混在一起的线性权重便不再匹配。

因此，要构造 $O(3)$ 等变卷积层，核心原则是：**严格区分标量、极矢量与轴矢量三类通道，绝不直接混合不同阶或不同宇称的特征；在跨阶或同阶交流信息时，借助网格邻居之间的相对方向向量，将源特征经几何运算转换为目标阶数与目标宇称，再施加仅依赖距离的各向同性径向权重。**

### 2.2 利用相对方向向量实现跨阶与同阶信息传递

考虑三维周期卷积核内任意整数偏移 $\Delta\mathbf j \in \{-r_{\max}, \dots, r_{\max}\}^3$（核宽 $K = 2r_{\max}+1$ 为奇数）。将其拆分为标量欧氏距离 $r = \|\Delta\mathbf j\|_2$ 与单位方向向量：

$$
\mathbf n(\Delta\mathbf j) = \begin{cases}
\mathbf 0, & \Delta\mathbf j = \mathbf 0, \\
\dfrac{\Delta\mathbf j}{\|\Delta\mathbf j\|_2}, & \Delta\mathbf j \neq \mathbf 0.
\end{cases}
$$

由于距离 $r$ 在 $O(3)$ 下不变，而方向向量 $\mathbf n$ 是极矢量（$\mathbf n' = R\mathbf n$），邻居特征通过与 $\mathbf n$ 的几何组合即可安全地在不同阶数之间传递信息。逐类分析本项目中的全部耦合路径如下：

#### 1. 标量特征 $s$ 向外贡献信息
- **贡献给标量（直接加权）**：乘以仅依赖距离的径向权重 $R_{\rm s}(r)\,s$，仍为标量。
- **贡献给极矢量（乘以方向向量）**：用方向向量将标量升阶为极矢量 $R_{\rm grad}(r)\,s\,\mathbf n$（离散梯度形式）。在 $R \in O(3)$ 下变换为 $s\,(R\mathbf n) = R(s\mathbf n)$，严格满足极矢量变换律。
- **贡献给轴矢量（无直接路径）**：标量 $s$ 与极矢量 $\mathbf n$ 相乘只能得到极矢量，无法在镜像反射下产生额外的 $\det R$ 因子，因此该耦合块为零。

#### 2. 极矢量特征 $\mathbf v$ 向外贡献信息
- **贡献给标量（与方向向量点积）**：构造内积 $R_{\rm div}(r)\,(\mathbf n \cdot \mathbf v) = R_{\rm div}(r)\,\mathbf n^\top \mathbf v$（离散散度形式）。验证等变性：

  $$
  (R\mathbf n)^\top (R\mathbf v) = \mathbf n^\top (R^\top R)\mathbf v = \mathbf n^\top \mathbf v,
  $$

  严格降阶为 $O(3)$ 不变标量。
- **贡献给极矢量（直接贡献 + 沿方向向量投影）**：
  - **直接贡献**：直接按径向权重缩放 $R_{\rm v}(r)\,\mathbf v = R_{\rm v}(r)\,I\mathbf v$，变换后为 $R(R_{\rm v}\mathbf v)$，保持极矢量性质。
  - **投影到方向向量**：将 $\mathbf v$ 投影到方向 $\mathbf n$ 上，得到 $(\mathbf n\cdot\mathbf v)\mathbf n = (\mathbf n\mathbf n^\top)\mathbf v$。验证其等变性：

    $$
    \big((R\mathbf n)(R\mathbf n)^\top\big)(R\mathbf v)
    = R\mathbf n(\mathbf n^\top R^\top R\mathbf v)
    = R\big((\mathbf n\mathbf n^\top)\mathbf v\big),
    $$

    同样严格保持极矢量性质。
  - **为何采用去均值形式 $\mathbf n\mathbf n^\top - \frac{1}{3}I$**：考察所有到中心距离同为 $r > 0$ 的等距邻居构成的壳层 $\mathcal S_r = \{\Delta\mathbf j : \|\Delta\mathbf j\|_2 = r\}$（共含 $M(r)$ 个格点）。由三维立方网格对坐标轴置换和正负号翻转的对称性：
    1. 对任意不同坐标轴 $i \neq j$，翻转第 $i$ 轴符号使 $n_i n_j \mapsto -n_i n_j$，正负成对抵消，故 $\sum_{\mathcal S_r} n_i n_j = 0$；
    2. 对三个对角分量，三轴完全等价且满足 $n_1^2 + n_2^2 + n_3^2 = \|\mathbf n\|_2^2 = 1$，故：

       $$
       \sum_{\mathcal S_r} n_1^2 = \sum_{\mathcal S_r} n_2^2 = \sum_{\mathcal S_r} n_3^2
       = \frac{1}{3}\sum_{\mathcal S_r}(n_1^2 + n_2^2 + n_3^2)
       = \frac{M(r)}{3}.
       $$

    合并非对角与对角项，得到壳层平均恒等式：

    $$
    \frac{1}{M(r)}\sum_{\Delta\mathbf j \in \mathcal S_r} \mathbf n\mathbf n^\top = \frac{1}{3}I.
    $$

    这意味着，若对均匀常矢量场 $\mathbf v$ 做方向投影并对等距邻居求平均，结果并不是零，而是 $\frac{1}{3}\mathbf v$——即原始投影 $\mathbf n\mathbf n^\top$ 中天然混入了 $\frac{1}{3}I$ 的“直接贡献”成分。若直接把 $I$ 与 $\mathbf n\mathbf n^\top$ 并列作为两条路径，二者会在各向同性分量上重叠耦合。

    因此，在非中心邻居上从 $\mathbf n\mathbf n^\top$ 中减去其壳层均值 $\frac{1}{3}I$，定义**无迹四极投影算子**：

    $$
    \boxed{
    Q(\mathbf n) = \begin{cases}
    \mathbf n\mathbf n^\top - \dfrac{1}{3}I, & \Delta\mathbf j \neq \mathbf 0, \\
    \mathbf 0, & \Delta\mathbf j = \mathbf 0.
    \end{cases}
    }
    $$

    一方面，$Q(\mathbf n)$ 依然严格满足极矢量等变条件 $Q(R\mathbf n)(R\mathbf v) = R\,Q(\mathbf n)\mathbf v$；另一方面，它满足 $\operatorname{tr} Q(\mathbf n) = 0$ 且 $\sum_{\mathcal S_r} Q(\mathbf n) = \mathbf 0$。这样，“直接贡献” $I\mathbf v$ 专门负责提取壳层的各向同性平均，而“去均值投影” $Q(\mathbf n)\mathbf v = (\mathbf n\cdot\mathbf v)\mathbf n - \frac{1}{3}\mathbf v$ 纯粹提取沿不同方向拉伸/压缩的各向异性差异（形式上恰好对应宇宙学中的无迹潮汐张量），两条路径完全解耦。

- **贡献给轴矢量（与方向向量叉积）**：构造 $\mathbf n \times \mathbf v$，写成矩阵形式为 $[\mathbf n]_\times \mathbf v$，其中反对称叉积矩阵为：

  $$
  [\mathbf n]_\times = \begin{pmatrix}
  0 & -n_z & n_y \\
  n_z & 0 & -n_x \\
  -n_y & n_x & 0
  \end{pmatrix}.
  $$

  由三维叉积在正交变换下的性质：

  $$
  (R\mathbf n) \times (R\mathbf v) = (\det R)\,R(\mathbf n \times \mathbf v).
  $$

  在纯旋转（$\det R = +1$）下它随空间同步旋转，但在镜像反射（$\det R = -1$）下会额外反号。因此 $\mathbf n \times \mathbf v$ 不是极矢量，而恰好是一个**轴矢量**。本项目在第一层卷积中正是通过这条路径，从输入动量场（极矢量）中自然生成轴矢量特征。

#### 3. 轴矢量特征 $\mathbf a$ 向外贡献信息
当网络进入第 2 层及更深层时，输入同时包含标量、极矢量与轴矢量：
- **贡献给标量（禁戒）**：点积 $\mathbf n \cdot \mathbf a$ 在 $R \in O(3)$ 下变换为 $(R\mathbf n)\cdot((\det R)R\mathbf a) = (\det R)(\mathbf n\cdot\mathbf a)$，镜像反射下会反号（是赝标量而非真标量），因此不能直接汇入标量通道，该耦合块为零。
- **贡献给极矢量（与方向向量叉积）**：计算 $\mathbf n \times \mathbf a = [\mathbf n]_\times \mathbf a$。检验其变换律：

  $$
  (R\mathbf n) \times \big((\det R)R\mathbf a\big)
  = (\det R)^2 R(\mathbf n \times \mathbf a)
  = R(\mathbf n \times \mathbf a),
  $$

  叉积引入的 $\det R$ 与轴矢量自带的 $\det R$ 相乘抵消（$(\det R)^2 = 1$），从而将轴矢量严格转换回极矢量。
- **贡献给轴矢量（直接贡献 + 无迹方向投影）**：与极矢量到极矢量完全对称，直接贡献 $R_{\rm a}(r)I\mathbf a$ 与无迹投影 $R_{\rm a,Q}(r)Q(\mathbf n)\mathbf a$ 均严格保持 $(\det R)R$ 的轴矢量变换律。

### 2.3 径向滤波器与完整等变卷积算子

为保证各向同性，卷积核中到中心距离相同的邻居共享同一径向权重。将核内所有互异的半径升序记为 $0 = r_0 < r_1 < \dots < r_S$，壳层格点数为 $M(r)$。在离散半径上定义高斯基函数：中心点 $r_0 = 0$ 单独取孤立基函数 $\phi_0(r) = \mathbb I(r=0)$，非中心壳层 $m = 1, \dots, S$ 取

$$
\phi_m(r) = \begin{cases}
0, & r = 0, \\
\exp\!\left(-\left(\dfrac{r - r_m}{\sigma}\right)^2\right), & r > 0,
\end{cases}
\qquad
\sigma = \min_{0 \le m < S}(r_{m+1} - r_m).
$$

对任意耦合路径 $\alpha$，经壳层点数归一化后的径向滤波器为：

$$
R_\alpha(r) = \frac{1}{M(r)}\sum_{m} W_{\alpha, m}\,\phi_m(r),
$$

其中 $W_{\alpha, m}$ 为可学习通道权重矩阵。各向同性路径（标量、极矢量与轴矢量的直接贡献）使用全部基函数 $m = 0, \dots, S$；所有依赖方向向量 $\mathbf n$ 的耦合路径在中心点 $\Delta\mathbf j = \mathbf 0$ 处为零，仅使用 $m = 1, \dots, S$。

汇总 2.2 节全部 9 条路径，周期三维等变卷积的线性映射可紧凑地写作 $3\times 3$ 分块算子：

$$
\boxed{
\begin{pmatrix}
\tilde s \\
\tilde{\mathbf v} \\
\tilde{\mathbf a}
\end{pmatrix}(\mathbf j)
= \sum_{\Delta\mathbf j}
\begin{pmatrix}
R_{\rm s}(r) & R_{\rm div}(r)\,\mathbf n^\top & \mathbf 0^\top \\
R_{\rm grad}(r)\,\mathbf n & R_{\rm v}(r)I + R_{\rm v,Q}(r)Q(\mathbf n) & R_{\rm a\to v}(r)\,[\mathbf n]_\times \\
\mathbf 0 & R_{\rm v\to a}(r)\,[\mathbf n]_\times & R_{\rm a}(r)I + R_{\rm a,Q}(r)Q(\mathbf n)
\end{pmatrix}
\begin{pmatrix}
s \\
\mathbf v \\
\mathbf a
\end{pmatrix}\big((\mathbf j + \Delta\mathbf j)\bmod N\big).
}
$$

第一层由于输入仅含密度标量场与动量极矢量场，算子自动取前两列（6 条路径），并由左下角 $R_{\rm v\to a}(r)[\mathbf n]_\times\mathbf v$ 生成轴矢量场供后续层使用。

### 2.4 标量门控非线性、残差连接与不变读出

1. **标量门控非线性**：对矢量分量逐个施加常规激活函数会破坏旋转等变性（一般地 $\sigma(R\mathbf v) \neq R\sigma(\mathbf v)$）。因此，仅对标量通道加上偏置 $\mathbf b_s$ 并计算 Sigmoid 门控系数 $g = \sigma(\tilde s + \mathbf b_s) \in (0, 1)$，再将该标量门统一乘到同通道的标量与三个矢量分量上：

   $$
   s_{\rm out} = (\tilde s + \mathbf b_s)\odot g, \qquad
   \mathbf v_{\rm out} = g\,\tilde{\mathbf v}, \qquad
   \mathbf a_{\rm out} = g\,\tilde{\mathbf a}.
   $$

   由于 $g$ 在 $O(3)$ 下不变，缩放操作严格满足 $g(R\tilde{\mathbf v}) = R(g\tilde{\mathbf v})$ 与 $g((\det R)R\tilde{\mathbf a}) = (\det R)R(g\tilde{\mathbf a})$，其中标量分支即标准的 SiLU 激活。当卷积层数大于 1 且启用残差连接时，从第 2 层起将同类型特征直接相加，同样保持严格等变性。

2. **全局 $O(3)$ 与平移不变读出**：经过 $L$ 层等变卷积得到 $C$ 通道的标量、极矢量与轴矢量特征场后，需要将其汇聚为与空间位置和朝向无关的标量向量，以输出步长概率与状态价值。对每个通道 $c \in \{1,\dots,C\}$ 计算：

   $$
   \bar s_c = \frac{1}{N^3}\sum_{\mathbf j} s_c(\mathbf j), \qquad
   \bar E_{v,c} = \frac{1}{N^3}\sum_{\mathbf j} \|\mathbf v_c(\mathbf j)\|_2^2, \qquad
   \bar E_{a,c} = \frac{1}{N^3}\sum_{\mathbf j} \|\mathbf a_c(\mathbf j)\|_2^2.
   $$

   利用正交变换保范性 $\|R\mathbf v\|_2^2 = \mathbf v^\top R^\top R\mathbf v = \|\mathbf v\|_2^2$ 与 $\|(\det R)R\mathbf a\|_2^2 = \|\mathbf a\|_2^2$，可知上述局部平方模的空间均值严格为 $O(3)$ 与平移不变量。将 $(\bar{\mathbf s}, \bar{\mathbf E}_v, \mathbf g_n, \bar{\mathbf E}_a) \in \mathbb R^{3C+2}$ 拼接后送入共享单隐藏层感知机（线性映射 + $\tanh$ 激活），再分别由线性策略头与线性价值头输出动作 logits 与状态价值 $V_\phi(o_n)$。

## 3. 科学误差度量、奖励设计与实验方案

### 3.1 带通误差度量

仅匹配功率谱无法保证宇宙学大尺度结构的空间相位一致，因此同时在可信傅里叶波数带内评估密度场逐模误差、球壳功率谱误差与交叉相关系数。定义带通截止波数：

$$
k_{\max} = 0.3\min\!\big(k_{{\rm Nyq,grid}},\; k_{{\rm Nyq,particle}}\big),
$$

记带通内有效非零半谱模式集合为 $\Omega = \{\mathbf q : 0 < |\mathbf q|/(2\pi) \le 0.3\min(N, N_p^{1/3})/2\}$。由于该带通远低于网格 Nyquist 频率，半谱内仅 $k_z = 0$ 平面自共轭，故定义共轭重数权重：

$$
w(\mathbf q) = \begin{cases}
1, & q_z = 0, \\
2, & q_z > 0.
\end{cases}
$$

对候选密度场 $\delta$ 与高分辨率参考密度场 $\delta_{\rm ref}$ 的前向归一化离散谱 $\widehat\delta(\mathbf q), \widehat\delta_{\rm ref}(\mathbf q)$，定义三项科学诊断量：

1. **带通相对密度场误差**：

   $$
   e_\delta = \frac{\sum_{\mathbf q \in \Omega} w(\mathbf q)\,|\widehat\delta(\mathbf q) - \widehat\delta_{\rm ref}(\mathbf q)|^2}{\sum_{\mathbf q \in \Omega} w(\mathbf q)\,|\widehat\delta_{\rm ref}(\mathbf q)|^2 + \epsilon}.
   $$

2. **球壳功率谱相对均方误差**：将 $\Omega$ 按无量纲波数整数分箱划分有效球壳 $\mathcal K$，每壳层共轭加权平均功率 $P(k) = \frac{\sum_{\mathbf q \in \text{shell } k} w(\mathbf q)|\widehat\delta(\mathbf q)|^2}{\sum_{\mathbf q \in \text{shell } k} w(\mathbf q)}$，定义：

   $$
   e_P = \frac{1}{|\mathcal K|}\sum_{k \in \mathcal K}\left[\frac{P(k) - P_{\rm ref}(k)}{P_{\rm ref}(k) + \epsilon}\right]^2.
   $$

3. **平均交叉相关系数**：反映相位一致性，

   $$
   \bar r = \frac{1}{|\mathcal K|}\sum_{k \in \mathcal K}
   \frac{\operatorname{Re} P_{\rm cross}(k)}{\sqrt{P(k)P_{\rm ref}(k)}}, \qquad
   P_{\rm cross}(k) = \frac{\sum_{\mathbf q \in \text{shell } k} w(\mathbf q)\,\widehat\delta(\mathbf q)\widehat\delta_{\rm ref}^*(\mathbf q)}{\sum_{\mathbf q \in \text{shell } k} w(\mathbf q)}.
   $$

### 3.2 有界损失与势能差分奖励

由于 $e_\delta, e_P$ 在极端步长下可能跨越多个数量级，先通过有理函数映射为 $[0, 1]$ 内的有界复合损失（默认等权 $w_\delta = w_P = 0.5$，尺度 $\tau_\delta = \tau_P = 10^{-6}$）：

$$
\mathcal L(a) = w_\delta\frac{e_\delta}{\tau_\delta + e_\delta} + w_P\frac{e_P}{\tau_P + e_P},
\qquad w_\delta, w_P \ge 0,\; w_\delta + w_P = 1,\; \tau_\delta, \tau_P > 0.
$$

本任务采用有限视界无折扣设置 $\gamma = 1$：不同动作序列对应不同的物理时间跨度与决策步数，若强行引入每步 $\gamma < 1$，会使相同终局精度仅因步数不同而获得不同回报，扭曲固定预算下的终局精度目标。

记初始损失 $\mathcal L(a_0) = 0$（候选解与参考解共享同一初值）。系统支持两种评分机制：仅终局评分，或每隔 $N_{\rm rew}$ 个动作及终局评分。在每次评分时刻 $t_k$，发放上次评分损失与本次损失之差：

$$
r_{t_k} = \mathcal L(a_{t_{k-1}}) - \mathcal L(a_{t_k}),
$$

非评分步奖励为 $0$；若因数值溢出或无合法动作失败终止，则以固定惩罚损失 $\mathcal L_{\rm fail} = 2$ 结算。在 $\gamma = 1$ 下，由裂项相消可知，无论中间评分频率如何，整条轨迹的累积回报严格等于负的最终损失：

$$
\sum_{n=0}^{T-1} r_n = \mathcal L(a_0) - \mathcal L_{\rm final} = -\mathcal L_{\rm final} \in [-1, 0] \quad (\text{成功}), \qquad -2 \quad (\text{失败}).
$$

中间评分时，参考解由同一初值按高分辨率固定对数步数（默认 $4B$ 步）生成：参考轨迹先推进至不大于当前尺度因子 $a_n$ 的最后一个固定参考节点，再临时向前探测一小步对齐至 $a_n$ 进行评分，且该临时探测步不改变后续固定参考轨迹的节点序列。

### 3.3 实验与评估方案

1. **参考解与基线设置**：给定预算 $B$（如 $B = 48$），参考解采用在 $\ln a$ 上均匀划分的 $4B$ 步固定步长 KDK 模拟；对比基线采用相同预算 $B$ 的均匀 $\ln a$ 固定步长 KDK 模拟。
2. **训练与泛化评估**：每次策略更新采集若干独立初始场种子的完整 episode；评估阶段在与训练集互不重叠的全新随机种子上，采用贪心策略（选取合法动作中 logit 最大者）运行，逐种子对比学习策略与同预算固定步长基线的总损失 $\mathcal L$、$e_\delta$、$e_P$、平均交叉相关系数 $\bar r$ 及实际消耗的 PM 步数。

## 4. GAE 优势估计与带掩码 PPO 目标

记第 $n$ 步观测为 $o_n$、完整状态为 $s_n$、所选动作索引为 $j_n$。优势函数 $A^\pi(s_n, j_n) = Q^\pi(s_n, j_n) - V^\pi(s_n)$ 衡量动作 $j_n$ 相对平均策略的增益，其估计面临**偏差与方差的权衡**。

### 4.1 从 MC、1-step TD 到 GAE 的完整推导

1. **蒙特卡洛（MC）估计**：以整条轨迹的实际回报 $G_n = \sum_{l=0}^{T-1-n}\gamma^l r_{n+l}$ 估计动作价值，得到 $\hat A_n^{\rm MC} = G_n - V_\phi(s_n)$。该回报估计无偏，但累积了后续全部随机转移与动作噪声，方差较高。
2. **单步时序差分（1-step TD）估计**：由贝尔曼方程 $Q^\pi(s_n, j_n) = \mathbb E[r_n + \gamma V^\pi(s_{n+1}) \mid s_n, j_n]$，用价值网络 $V_\phi$ 自举替代真实 $V^\pi$，得到单步 TD 残差：

   $$
   \hat A_n^{(1)} = \delta_n = r_n + \gamma V_\phi(s_{n+1}) - V_\phi(s_n).
   $$

   单步估计方差较低，但在 $V_\phi$ 尚未收敛准确时引入较大偏差。
3. **$k$-步优势估计的裂项展开**：向前展开 $k$ 步真实奖励并在第 $n+k$ 步以 $V_\phi(s_{n+k})$ 截断，构造 $k$-步优势 $\hat A_n^{(k)}$。逐项插入并减去中间时刻的 $\gamma^l V_\phi(s_{n+l})$：

   $$
   \begin{aligned}
   \hat A_n^{(1)} &= r_n + \gamma V_\phi(s_{n+1}) - V_\phi(s_n) = \delta_n, \\
   \hat A_n^{(2)} &= r_n + \gamma r_{n+1} + \gamma^2 V_\phi(s_{n+2}) - V_\phi(s_n) \\
   &= \big(r_n + \gamma V_\phi(s_{n+1}) - V_\phi(s_n)\big) + \gamma\big(r_{n+1} + \gamma V_\phi(s_{n+2}) - V_\phi(s_{n+1})\big) = \delta_n + \gamma\delta_{n+1}, \\
   &\;\;\vdots \\
   \hat A_n^{(k)} &= \sum_{l=0}^{k-1}\gamma^l r_{n+l} + \gamma^k V_\phi(s_{n+k}) - V_\phi(s_n) = \sum_{l=0}^{k-1}\gamma^l \delta_{n+l}.
   \end{aligned}
   $$

4. **广义优势估计（GAE）**：引入衰减因子 $\lambda \in [0, 1]$，对所有 $k$-步优势估计作指数加权平均（权重 $(1-\lambda)\lambda^{k-1}$ 满足 $\sum_{k=1}^\infty (1-\lambda)\lambda^{k-1} = 1$）：

   $$
   \begin{aligned}
   \hat A_n^{\rm GAE}
   &= (1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\hat A_n^{(k)} \\
   &= (1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\sum_{l=0}^{k-1}\gamma^l\delta_{n+l} \\
   &= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\delta_{n+l}\sum_{k=l+1}^{\infty}\lambda^{k-1} \\
   &= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\delta_{n+l}\frac{\lambda^l}{1-\lambda}
   = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{n+l}.
   \end{aligned}
   $$

对有限长度 $T$ 的完整 episode（在终止态 $s_T$ 严格无后续回报，取 $V_\phi(s_T) = 0$），截断至 $T-1$ 并代入本项目设定 $\gamma = 1$（默认 $\lambda = 0.95$），得到最终有限视界 GAE 公式及 $O(T)$ 逆向递推关系：

$$
\boxed{
\hat A_n^{\rm GAE} = \sum_{l=0}^{T-1-n}\lambda^l \delta_{n+l}, \qquad
\hat A_n^{\rm GAE} = \delta_n + \lambda\hat A_{n+1}^{\rm GAE}, \quad \hat A_T^{\rm GAE} = 0.
}
$$

- 当 $\lambda = 0$ 时，退化为单步 TD 估计 $\delta_n$；
- 当 $\lambda = 1$ 时，中间价值项完全裂项相消，退化为无偏蒙特卡洛估计 $G_n - V_\phi(s_n)$；
- 当 $0 < \lambda < 1$ 时，平滑折中偏差与方差。

价值网络的回归目标同步定义为 $\hat R_n = \hat A_n^{\rm GAE} + V_\phi^{\rm old}(o_n)$。

### 4.2 PPO 联合优化目标

在每次策略更新中，先对本批所有样本的优势做标准化 $\tilde A_n = \frac{\hat A_n - \mu_A}{\sigma_A + 10^{-8}}$。定义新旧带掩码策略的重要性采样比率：

$$
\rho_n(\theta) = \exp\!\Big(\ln\pi_\theta(j_n \mid o_n) - \ln\pi_{\theta_{\rm old}}(j_n \mid o_n)\Big).
$$

最小化如下三项加权组成的 PPO 联合损失函数：

$$
\boxed{
\mathcal J_{\rm PPO}(\theta, \phi)
= \mathcal L_{\rm clip}(\theta)
+ c_1 \mathcal L_{\rm VF}(\phi)
- c_2 \mathbb E_n\!\big[H(\pi_\theta(\cdot \mid o_n))\big],
}
$$

其中：
- **裁剪代理策略损失**（默认裁剪阈值 $\epsilon = 0.2$）：

  $$
  \mathcal L_{\rm clip}(\theta) = -\mathbb E_n\!\Big[\min\!\big(\rho_n(\theta)\tilde A_n,\; \operatorname{clip}(\rho_n(\theta), 1-\epsilon, 1+\epsilon)\tilde A_n\big)\Big];
  $$

- **价值函数均方误差损失**（默认权重 $c_1 = 0.5$）：

  $$
  \mathcal L_{\rm VF}(\phi) = \mathbb E_n\!\Big[\big(V_\phi(o_n) - \hat R_n\big)^2\Big];
  $$

- **策略熵正则项**（默认权重 $c_2 = 0.01$）：鼓励在合法动作子空间内保持探索。

优化器采用 Adam（默认学习率 $3\times 10^{-4}$），每批轨迹执行多轮小批量随机梯度更新（默认 4 轮、小批量大小 8），并对全局梯度施加 $L_2$ 范数裁剪（阈值 $0.5$）以保证训练稳定性。

## 参考文献

- Feng, Y., Chu, M.-Y., Seljak, U., & McDonald, P. (2016). *FastPM: a new scheme for fast simulations of dark matter and haloes*. MNRAS, 463(3), 2273–2286.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Unke, O. T., & Maennel, H. (2024). *E3x: E(3)-Equivariant Deep Learning Made Easy*. arXiv:2401.07595.
