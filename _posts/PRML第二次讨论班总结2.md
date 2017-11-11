---
layout: post
title: What's PRML?
---
#第二次PRML讨论班总结
10月29日，本学期第二次PRML讨论班顺利举办。本期的主题是线性回归模型，主讲人为来自中科院计算所的闵越聪同学。本次报告从频率学派和贝叶斯学派两个不同角度分别对线性回归模型进行了分析。

##频率学派
###1.基础线性模型
在线性回归问题中，最简单的线性模型为：
$$y(\mathbf{x},\mathbf{w})=\mathbf{w}^\mathrm{T}\mathbf{x}=w_0+w_1x_1+\cdots+w_Dx_D$$

其中 $\mathbf{x}$为输入数据，$\mathbf{w}$为所求线性模型参数。
将上述模型进行扩展，$\mathbf{x}$可由多个$\mathbf{x}$的函数取代：
$$
y(\mathbf{x},\mathbf{w}) = \mathbf{w}^\mathrm{T}\phi(\mathbf{x})= w_0+\sum_{j=1}^{M-1}w_j \phi_j(\mathbf{x})
$$
其中$\phi$被称作线性模型的基方程(例如高斯函数、sigmoid函数等非线性函数，这样模型就进化成了非线性模型)，$\mathbf{w}$为模型参数
###2.损失函数
针对每一个数据$\mathbf{x}$与其目标$t$，回归估计为$y(\mathbf{x})$，我们希望使得回归总误差最小：
$$\min\quad E[L]=\iint L(t,y(\mathbf{x}))p(\mathbf{x},t)dxdt$$
其中$ L(t,y(\mathbf{x}))$为损失函数。举个栗子，我们可以选取误差平方$L(t,y(\mathbf{x}))=(y(\mathbf{x})-t)^2$作为损失函数，则回归总误差可表示为：
$$\min\quad E[L]=\iint (y(\mathbf{x})-t)^2p(\mathbf{x},t)d\mathbf{x}dt$$

对$ E[L]$计算$y(\mathbf{x})$偏导数得
$$\frac{\delta E[L]}{\delta y(\mathbf{x})}=2\int (y(\mathbf{x})-t)p(\mathbf{x},t)d\mathbf{x}$$

令上式等于0，计算$y(\mathbf{x})$得：
$$y(\mathbf{x})=\frac{\int tp(\mathbf{x},t)dt}{p(\mathbf{x})}=\int tp(t|\mathbf{x})dt=E_t[t|\mathbf{x}]$$

可见此时回归方程等价于$t$在$\mathbf{x}$确定时的条件概率下的期望：
![http://i67.tinypic.com/2dvq350.jpg](http://i67.tinypic.com/2dvq350.jpg)
其中蓝线可以理解成$t$在$\mathbf{x}$确定时的条件概率，而$t$的期望正是$y(\mathbf{x}_0)$，按照这样的规则可求得回归方程（红线）。
此外，我们可以将平方项分解：

$$\begin{aligned}(y(\mathbf{x})-t)^2=&(y(\mathbf{x})-E_t[t|\mathbf{x}]+E_t[t|\mathbf{x}]-t)\newline=&(y(\mathbf{x})-E_t[t|\mathbf{x}])^2+(E_t[t|\mathbf{x}]-t)^2\newline &+2(y(\mathbf{x})-E_t[t|\mathbf{x}])(E_t[t|\mathbf{x}]-t)\end{aligned}$$

将上式代入总误差函数得：
$$\begin{aligned}E[L]=&\int (y(\mathbf{x})-E_t[t|\mathbf{x}])^2p(\mathbf{x})dx+\iint (E_t[t|\mathbf{x}]-t)^2p(\mathbf{x},t)d\mathbf{x}dt\end{aligned}$$

可见总误差函数能够分解成回归函数$y(\mathbf{x})$和目标条件分布期望的误差，以及目标条件分布期望与真实目标值的误差两大部分。
###3.最大似然和最小二乘
接下来我们从最大似然角度对线性回归进行分析。首先假设目标$t$由某个确定的函数$y(\mathbf{x},\mathbf{w})$加上高斯噪声$\epsilon \sim \mathcal{N}(0,\beta^\mathrm{-1})$得到：

$$t=y(\mathbf{x},\mathbf{w})+\epsilon$$

因此$t$的条件概率可写作：
$$p(\mathbf{t}|\mathbf{x},\mathbf{w},\beta) = \mathcal{N}(t|y(\mathbf{x},\mathbf{w}),\beta^\mathrm{-1})$$

条件概率期望简化得：
$$E[t|x]=\int tp(t|\mathbf{x})dt=y(\mathbf{x},\mathbf{w})$$

考虑数据集$\mathrm{X}=\{x_1,\cdots,x_N\}$及其目标$t_1,\cdots,t_N$，对数似然函数为：

$$\begin{aligned}\ln p(\mathbf{t}|\mathbf{X},\mathbf{w},\beta)&=\ln(\prod_{n=1}^N\mathcal{N}(t_n|\mathbf{w}^\mathrm{T}\bf{\phi}(x_n),\beta^\mathrm{-1}))\\&=\frac{N}{2}\ln \beta-\frac{N}{2}\ln(2\pi)-\beta E_D(\mathbf{w})\\&= const -\beta E_D(\mathbf{w})\end{aligned}$$

其中误差平方和定义为：

$$	E_D(\mathbf{w})=\frac{1}{2}\sum^N_{n=1}\{t_n-\mathbf{w}^\mathrm{T}\bf{\phi}(x_n)\}^2$$

对对数似然函数求导：

$$\nabla \ln p(\mathbf{t}|\mathbf{X},\mathbf{w},\beta)=\sum^N_{n=1}\{t_n-\mathbf{w}^\mathrm{T}\bf{\phi}(x_n)\}\bf{\phi}(x_n)^\mathrm{T}$$

令导数为0，求解$\mathbf{w}$:

$$\begin{aligned}0 &= \sum^N_{n=1}t_n \bf{\phi}(x_n)^\mathrm{T}-\mathbf{w}^\mathrm{T}( \sum^N_{n=1}\bf{\phi}(x_n) \bf{\phi}(x_n)^\mathrm{T})\\0&=\Phi^\mathrm{T}\mathbf{t}-\Phi^\mathrm{T}\Phi \mathbf{w}\end{aligned}$$

$$w_{ML}=(\Phi^\mathrm{T}\Phi)^\mathrm{-1}\Phi^\mathrm{T}\mathbf{t}$$

熟悉最小二乘的同学一定能够发现，上式最大似然估计的解和最小二乘的解的式子是完全相同的！此外，我们还能用同样的方法解得其他参数：

$$	w_0=\bar{t}-\sum_{j=1}^{M-1}w_j\bar{\phi_j}$$

其中$\bar{t}=\frac{1}{N}\sum_{n=1}^{N}t_n,\qquad \bar{\phi_j}=\frac{1}{N}\sum_{n=1}^N \phi_j(x_n)$

以及对误差参数$\beta$计算得：

$$	\frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^N\{t_n-\mathbf{w}_{ML}^T\bf{\phi}(x_n)\}^2$$

###4.最小二乘法的几何解释

![](http://i64.tinypic.com/30mbu3l.jpg)

线性拟合可以理解成线性求解问题，定义$$\Phi=\left(\begin{array}{cccc}\phi_1(x_1) & \phi_2(x_1) & ... & \phi_n(x_1)\\\phi_1(x_2) & \phi_2(x_2) & ... & \phi_n(x_2)\\... & ... & ... & ...\\\phi_1(x_m) & \phi_2(x_m) & ... & \phi_n(x_m)\end{array}\right)$$

线性回归理想情况下满足于等式$\Phi \mathbf{w}=t$，但是实际情况下等式不成立，即$t$不在$\Phi$的解空间内，不存在线性组合$\mathbf{w}$使得等式成立。此时我们退而求其次，寻找$\hat{\mathbf{w}}$使得$\Phi\hat{\mathbf{w}}$所表示的向量最接近目标向量$t$，而这个向量就是$t$在$\Phi$解空间$S$上的投影（如图所示，蓝色向量$y$即为所求投影)，利用解空间$S$与目标向量和投影向量的残差向量正交的性质可得公式：

$$\Phi^T(t-\Phi\hat{\mathbf{w}})=0$$

易解得：

$$\hat{\mathbf{w}}=(\Phi^\mathrm{T}\Phi)^\mathrm{-1}\Phi^\mathrm{T}\mathbf{t}$$

对比之前最大似然估计的解可以发现两者一致

### 5.序列学习

由于回归的总体误差包括了各个样本点回归的误差，因此在训练模型时可以利用梯度下降算法快速收敛求解$\mathbf{w}$:

$$\begin{aligned}\mathbf{w}^{(\tau+1)}=&\mathbf{w}^{(\tau)}-\eta\nabla E_n\\\mathbf{w}^{(\tau+1)}=&\mathbf{w}^{(\tau)}-\eta(t_n-\mathbf{w}^{(\tau)\mathrm{T}}\bf{\phi}_n)\bf{\phi}_n\end{aligned}$$

![IMG](http://i68.tinypic.com/2lbywkh.jpg)

### 6.正则化

拟合时的参数数量选取是一门艺术，过少的参数难以拟合复杂曲线，而过多的参数很容易造成过拟合的现象。为了避免过拟合的现象发生，我们在已有误差函数基础上引入正则化项(这项技术在机器学习领域也被叫做权重衰减weight decay)：

$$E_D(\mathbf{w})+\lambda E_W(\mathbf{w})$$

一个最简单的正则化项可以设为：

$E_W(\mathbf{w})=\frac{1}{2}\mathbf{w}^\mathrm{T}\mathbf{w}$

当选取平方和误差作为误差函数时：

$$E_D(\mathbf{w})=\frac{1}{2}\sum^N_{n=1}\{t_n-\mathbf{w}^\mathrm{T}\bf{\phi}(x_n)\}^2$$

对带有正则化项的误差函数进行求解，可以解得：

$$\mathbf{w} = (\lambda I + \Phi^\mathrm{T}\Phi)^\mathrm{-1}\Phi t$$

此外一个更加通用的正则化模型表示如下：

$$\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^\mathrm{T}\bf{\phi}(x_n)\}^2+\frac{\lambda}{2}\sum_{j=1}^M|w_j|^q$$

或者也可以把上式写成求最值形式：

$$\begin{aligned}\min \quad &\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^\mathrm{T}\bf{\phi}(x_n)\}^2\\s.t. \quad &\sum^M_{j=1}|w_j|^q\leq \eta\end{aligned}$$

在正则化项中，$q$的取值至关重要，会对抑制过拟合的效果产生重要影响。接下来我们来分析当$q=1$和$q=2$时正则化效果：

![](http://i68.tinypic.com/1o8w8o.jpg)图一![](http://i63.tinypic.com/2d9bmmc.jpg)图二

上图红色部分代表了$\sum^M_{j=1}|w_j|^q\leq \eta$约束条件下$\mathbf{w}$的取值可行区域，图一$q=1$，图二$q=2$；蓝色区域中心则表示最优解。比较两幅图可以看出当正则项为$L_1$范数时，可行域内最优解会落在坐标轴上，这就意味这$L_1$范数的解是稀疏的。而$L_2$范数作为正则化项的最优解并不是稀疏的。

### 7.偏差方差分解

让我们回到第2节讲到的误差函数：

$$\begin{aligned}E[L]&=\int \{y(\mathbf{x})-h(\mathbf{x})\}^2p(\mathbf{x})d\mathbf{x}+\iint \{h(\mathbf{x})-t\}^2p(\mathbf{x},t)d\mathbf{x}dt \\h(\mathbf{x})&=E_t[t|\mathbf{x}]=\int tp(t|\mathbf{x})dt\end{aligned}$$

我们分析过，误差函数的第一部分的大小取决于我们对损失函数的选择，第二部分的大小取决于训练数据固有的扰动。接下来我们针对某个特定数据集$D$对第一项进行继续分解，在式子中间加上并减去$E_D[y(x;D)]$：

$$\begin{aligned}&\{y(\mathbf{x};D)-E_D[y(\mathbf{x};D)]+E_D[y(\mathbf{x};D)]-h(\mathbf{x})\}^2 \\	=&\{y(\mathbf{x};D)-E_D[y(\mathbf{x};D)]\}^2+\{E_D[y(\mathbf{x};D)]-h(\mathbf{x})\}^2\\	&+2\{y(\mathbf{x};D)-E_D[y(\mathbf{x};D)]\}\{E_D[y(\mathbf{x};D)]-h(\mathbf{x})\}\end{aligned}$$

因此可得定义偏差方差分解：

$$\begin{aligned}&E_D[\{y(\mathbf{x};D)-h(\mathbf{x})\}^2]\\=&\underbrace{	    E_D[\{y(\mathbf{x};D)-E_D[y(\mathbf{x};D)]\}^2]}_{\text{ variance }}+ 		\underbrace{\{E_D[y(\mathbf{x};D)]-h(\mathbf{x})\}^2}_{(\text{bias})^2}\end{aligned}$$

所以误差的期望可以表示为$loss=(bias)^2+variance+noise$

![](http://i67.tinypic.com/32zrfid.jpg)

结合图例可知，当模型过于简单时（正则化系数较大），从数据集中选取一部分样本点进行拟合，多次拟合之后（如第一行红色曲线）会发现得到的多个模型之间的方差部分较小，而偏差较大（第二行图片）。随着模型复杂度的加强（正则化系数减小），不同样本点训练的模型之间方差变大，而平均之后与真实函数偏差较小。方差、偏差的关系如下图。综上，正则化参数的选择是一门艺术。

![](http://i65.tinypic.com/2cwqyd1.jpg)

##贝叶斯线性回归

###1.参数分布

从贝叶斯理论出发，任何参数都满足概率分布：

$$p(\mathbf{w}|\mathcal{D})=\frac{(\mathcal{D}|\mathbf{w})p(\mathbf{w})}{p(\mathcal{D})}$$

参数、输入固定下$t$的条件概率可以写作：

$$	p(\mathbf{t}|\mathbf{x},\mathbf{w},\beta) = \mathcal{N}(\mathbf{t}|y(\mathbf{x},\mathbf{w}),\beta^\mathrm{-1})$$

记参数$\mathbf{w}$的先验分布：

$$	p(\mathbf{w})= \mathcal{N}(\mathbf{w}|m_0,S_0)$$

得参数$\mathbf{w}$的后验分布：

$$\begin{aligned}p(\mathbf{w}|\mathbf{t})&= \mathcal{N}(\mathbf{w}|m_N,S_N)\\where\quad m_N&=S_N(S_0^\mathrm{-1}m_0+\beta\Phi^\mathrm{T}\mathbf{t})\\S_N^\mathrm{-1} &= S_0^\mathrm{-1}+\beta\Phi^\mathrm{T}\Phi\end{aligned}$$

为了方便起见，假定先验分布概率满足零均值各向同性高斯分布，这个分布由一个精度参数$\alpha$控制：

$$p(\mathbf{w})= \mathcal{N}(w|0,\alpha^\mathrm{-1}I)$$

对应的后验概率分布中：

$$\begin{aligned}m_N&=\beta S_N\Phi^\mathrm{T}\mathbf{t}\\S_N^\mathrm{-1} &= \alpha I+\beta\Phi^\mathrm{T}\Phi\end{aligned}$$

后验概率分布的对数由对数似然函数与先验的对数求和的方式得到，满足如下形式：

$$	\ln p(\mathbf{w}|\mathbf{t}) = -\frac{\beta}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^\mathrm{T}\phi(x_n)\}^2-\frac{\alpha}{2}\mathbf{w}^\mathrm{T}\mathbf{w}+const$$

于是，后验分布关于$w$的最大化等价于对平方和误差函数加上一个二次正则项进行最小化。

让我们结合图片来分析基于贝叶斯理论的迭代算法：

![](http://i67.tinypic.com/ev81v8.jpg)

这张图的第一行对应于观测到任何数据点之前的情况，给出了$w$空间的先验概率分布的图像，以及函数$y(x,w)$的六个样本，这六个样本的$w$都是从先验概率分布中抽取的。在第二行，我们看到了观测到一个数据点之后的情形。数据点的位置$(x,t)$由右侧一列中的蓝色圆圈表示。左侧一列是对于这个数据点的似然函数$p(t,w)$关于$w$的函数图像。注意，似然函数提供了一个温和的限制，即直线必须穿过数据点附近的位置，其中附近位置的范围由噪声精度$\beta$确定。为了进行对比，用来生成数据集的真实参数值在上图左侧一列被标记为白色十字。如果我们把这个似然函数与第一行的先验概率相乘，然后归一化，我们就得到了第二行中间的图给出的后验概率分布。从这个后验概率分布中抽取$w$的样本，对应的回归函数$y(x,w)$被花在了右侧一列的图中。注意，这些样本直线全部穿过数据点的附近位置。这张图的第三行展示了观测到第二个数据点的效果。与之前一样，这个数据点由右侧一列的蓝色圆圈表示。第二个数据点自身对应的似然函数在左侧一列的图中给出。如果我们把这个似然函数与第二行的后验概率分布相乘，我们就得到了第三行中间一列的图给出的后验概率分布。注意，这个后验概率分布与我们将原始的先验分布结合两个数据点的似然函数得到的后验概率分布完全相同。现在，后验概率分布被两个数据点影响。由于两个点足够定义一条直线，因此目前已经得到了相对较好的后验概率分布。从这个后验分布中抽取的样本产生了第三列中红色的函数，我们看到这些函数同时穿过了两个数据点的附近。第四行展示了观测到20个数据点的效果。左侧的图展示了第20个数据点自身的似然函数，中间的图展示了融合了20次观测信息的后验概率分布。注意与第三行相比，这个后验概率分布变得更加尖锐。在无穷多个数据点的极限情况下，后验概率分布会变成一个Delta函数。这个函数的中心是用白色十字标记出的真实参数值。

### 2.预测分布

得到后验概率之后，我们可以计算预测分布：

$$	p(t|\mathbf{t},\alpha,\beta)=\int p(t|\mathbf{w},\beta)p(\mathbf{w}|\mathbf{t},\alpha,\beta)d\mathbf{w}$$

其中$\mathbf{t}$是训练数据的目标变量的值组成的向量。可以进一步解得：

$$\begin{aligned}p(t|\mathbf{x},\mathbf{t},\alpha,\beta)=\mathcal{N}(t|m_N^\mathrm{T}\bf\phi(\mathbf{x}),\sigma^2_N(\mathbf{x}))\\where\quad 	\sigma^2_N(\mathbf{x})=\frac{1}{\beta}+\bf\phi(\mathbf{x})^\mathrm{T}S_N\bf\phi(\mathbf{x})\end{aligned}$$

为了说明贝叶斯线性回归模型的预测分布，让我们回到第1.1节...成的正弦数据集。在图3.8中，我们调整一个由高斯基函数线性组合的模型，使其适应于不同规模的数据集，然后观察对应的后验概率分布。这里，绿色曲线对应着产生数据点的函数$\sin(2\pi x)$（带有附加的高斯噪声）。大小为N = 1，N = 2，N = 4和N = 25的数据集在四幅图中用蓝色圆圈表示。对于每幅图，红色曲线是对应的高斯预测分布的均值，红色阴影区域是均值两侧的一个标准差范围的区域。注意，预测的不确定性依赖于$x$，并且在数据点的邻域内最小。还要注意，不确定性的程度随着观测到的数据点的增多.逐渐减小。

![](http://i68.tinypic.com/15ib29k.jpg)![](http://i63.tinypic.com/10e3jv6.jpg)

### 3.等价核

前文提到的线性基函数模型的后验均值解有一个有趣的解释，这个解释为和方法提供了舞台。我们可以把预测均值改写成如下形式：

$$y(\mathbf{x},m_N) = m_N^T \bf\phi(\mathbf{x})=\beta \bf\phi(\mathbf{x})^\mathrm{T}S_N\Phi^\mathrm{T}\mathbf{t}=\sum_{n=1}^N \beta \bf\phi(\mathbf{x})^\mathrm{T}S_N \bf\phi(x_n)t_n$$

预测分布的均值在某点$x$可以由训练集的目标变量$t_n$的线性组合给出：

$$y(\mathbf{x},m_N)=\sum_{n=1}^Nk(\mathbf{x},x_n)t_n$$

其中，函数

$$ k(\mathbf{x},x_n)=\beta\phi(\mathbf{x})^\mathrm{T}S_N \phi(x_n)$$

被称作平滑矩阵或等价核。像这样的回归函数，通过对训练集里目标值进行线性组合做预测，被称为线性平滑。下图给出了高斯基函数的情形下的等价核。图中给出了三个不同的$x$值的情况下，核函数$k(x,x')$与$x'$的函数关系。我们看到，它们局限在$x$的周围，因此在$x$处的预测分布的均值$y(x,m_N)$可以通过对目标值加权组合的方式获得。距离$x$较近的数据点可以赋一个较高的权值，而距离$x$较远的数据点可以赋一个较低的权值。直观来看，与远处的证据相比，我们把局部的证据赋予更高的权值似乎是更合理的。



![](http://i64.tinypic.com/10sdklx.jpg)

我们还可以获得更多的关于等价核的认识。考虑$y(x)$和$y(x')$的协方差：

$$\begin{aligned}cov[y(\mathbf{x}),y(x_n)]&=cov[\bf\phi(\mathbf{x})^\mathrm{T}\mathbf{w},\mathbf{w}^\mathrm{T}\bf\phi(x_n)]\\&=\bf\phi(\mathbf{x})^\mathrm{T}S_N \bf\phi(x_n)\\&=\beta^\mathrm{-1}k(\mathbf{x},x_n)\end{aligned}$$

根据等价核的形式，我们可以看到在附近的点处的预测均值相关性高，而对于距离较远的点对，相关性就较低。