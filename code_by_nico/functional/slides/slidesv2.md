---
title: Scalar-on-Function Bayesian Quantile Regression with Heteroskedastic Data
author: Nicolas Escobar 
header-includes:
  - \usepackage{listings}
  - \usepackage{tikz}
---
<!-- 
# Project 1: BQR for Scalar-on-Function Regression with Heteroskedastic Data -->

## Background 

### Linear quantile regression 

Simple setting: $P[Y_i \leq y] = F(y | X_i)$, scalars:

- Choose a quantile $0 < \tau < 1$
- Assume $Q_\tau(Y_i | X_i) = \beta_0 + \beta_1 X_i$ where $P[Y_i \leq Q_\tau(Y_i|X_i) | X_i] = \tau$
- Estimate $\beta_0$, $\beta_1$ by 
    \begin{equation*}
    (\hat \beta_0, \hat \beta_1) = \mathrm{argmin}_{\beta_0, \beta_1} \sum_i \rho_\tau(y_i - \beta_0 + \beta_1 x_i)
    \end{equation*}
    where $\rho_\tau(u) = (\tau - I(u < 0))u$

### Bayesian Quantile Regression

- At first sight, a Bayesian approach to quantile regression is puzzling, because there is no likelihood
- Asymmetric Laplace (AL) distribution:
  \begin{equation*}
  \log f_\tau(y | \mu, \sigma) \propto -\rho_\tau\left(\frac{y - \mu}{\sigma}\right)
  \end{equation*}
- Bayesian Quantile Regression (BQR): We don't know $F$, but we use the model $f_Y(y) = f_\tau(y | \beta_{0, \tau} + \beta_{1, \tau} x_i, \sigma)$
- Equivalently $Y_i = \beta_{0,\tau} + \beta_{1, \tau} x_i + \epsilon_i$, with $\epsilon_i \sim f_\tau(\cdot| 0, \sigma)$
- We perform Bayesian inference on $\beta_{0, \tau}, \beta_{1, \tau}, \sigma$, usually with MCMC

### GAL distribution 

- AL distribution is too rigid. For instance, it's always symmetric for median regression
- AL admits the mixture representation 
  \begin{equation*}
  \epsilon = A\nu + u \sqrt{\sigma B \nu}
  \end{equation*}
  where $A, B, \sigma$ are constants, $\nu \sim \mathrm{Exp}(1)$ and $u \sim \mathrm{N}(0,1)$
- Yan and Kottas (2017) propose the following generalization:
  \begin{equation*}
  \epsilon = \alpha s + A\nu + u \sqrt{\sigma B \nu}
  \end{equation*}
  where $\alpha$ is a constant, $s \sim \mathrm{N}^+(0,1)$
- This is called the Generalized Asymmetric Laplace (GAL) distribution. It can be reparametrized in terms of $\tau, \mu, \sigma, \gamma$.
- Thus, the model becomes $Y_i = \beta_{0, \tau} + \beta_{1,\tau} x_i + \epsilon_i$ with $\epsilon_i \sim \mathrm{GAL}_\tau(0, \sigma, \gamma)$

### Hamiltonian Monte Carlo {.fragile}

- HMC has gained popularity as an alternative to Gibbs, Metropolis
- Stan programming language 
- \lstinline!rstan! implements Stan in R
- \lstinline!brms! uses \lstinline!rstan! to implement formula syntax (as in \lstinline!lme4!) and nonparametric capabilities (with \lstinline!mgcv!)

## Our problem

### Low dimensional ilustration 

-  Consider data generated according to  
    \begin{equation*}
    y_i = x_i^2 + \delta_i
    \end{equation*}
    where $\delta_i \sim N(0, \exp(4x_i))$

![Heteroskedastic data](code_by_nico/functional/slides/htskdata.png){width=50%}

### Warning {.fragile}

- Quantile regression should be able to handle this kind of heteroskedasticity 
- It turns out you have to be careful:
    \begin{lstlisting}
    model1 <- brm(y ~ s(x), family = 'GAL')
    model2 <- brm(y ~ s(x), log sigma ~ s(x), 
                    family = 'GAL')
    \end{lstlisting}

![With and Without $\sigma$](code_by_nico/functional/slides/wandwo.png){width=50%}

### Literature{.fragile}

- The failure of \lstinline!model1! to model heteroskedastic data has gone unnoticed in the literature
- It has been recognized that GAL is still not flexible enough 
  - Dirichlet's mixtures of GAL distributions have become popular:
    $\mathrm{MGAL}_\tau = \sum_k \pi_k \mathrm{GAL}_\tau(0, \sigma_k, \gamma_k)$
- Two problems:
  <!-- - Ammounts to 
    \begin{lstlisting}
    model1 <- brm(y ~ s(x), family = 'MGAL')
    \end{lstlisting}
    not adressing the heteroskedasticity directly  -->
  - Computationally challenging
  - Blunt

### Framework

- Consider data generated as 
  \begin{align*}
  Y_i &= \beta_0 + \mathbf{Z}_i^T \boldsymbol{\beta}_z + \int_I \mathbf{X}_i(t)^T \boldsymbol{\beta}_{1}(t) dt \\ &\qquad + \exp\left(\eta_0 + \mathbf{Z}_i^T \boldsymbol{\eta}_z + \int_I \mathbf{X}_i(t)^T \boldsymbol{\eta}_{1}(t) dt\right) \delta_i
  \end{align*}
- Exponential heteroskedasticity. 
- No distributional assumption on $\delta_i$, other than iid.
- Easy to see that 
  \begin{align*}
  Q_\tau(Y_i|\mathbf{Z}_i, X_i ) &= \beta_{0} + \mathbf{Z}_i^T \boldsymbol{\beta}_{z} + \int_I \mathbf{X}_i(t)^T \boldsymbol{\beta}_{1}(t) dt \\
  &\quad + \exp\left(\eta_0 + \mathbf{Z}_i^T \boldsymbol{\eta}_z + \int_I \mathbf{X}_i(t)^T \boldsymbol{\eta}_{1}(t) dt\right) q_\tau
  \end{align*}
  $q_\tau$ being the $\tau$ quantile of $\delta_i$
<!-- - So, quantiles are still linear, the problem is still tractable. -->

### Basis expansion

- Write 
  \begin{equation*}
  \mathbf{X}_i = \sum_{k = 1}^K X_{i,k} \mathbf{b}_{k}
  \end{equation*}
  where the $\mathbf{b}_{k}$'s are some set of basis functions
- Denote 
  \begin{align*}
  \tilde{\mathbf{X}}_i &= (X_{i,1}, \ldots, X_{i, K})^T \\
  \tilde \beta_{1, k} &= \int_I \mathbf{b}_k(t)^T \boldsymbol \beta_{1}(t) dt \\
  \tilde{\boldsymbol\beta}_{1} &= (\tilde\beta_{1,1},\ldots, \tilde\beta_{1,K})^T
  \end{align*} 
  Define $\tilde{\boldsymbol{\eta}}_1$ similarly

### Approach 

- We try to estimate
  \begin{align*}
  Q_\tau(Y_i|\mathbf{Z}_i, X_i ) &= \beta_{0} + \mathbf{Z}_i^T \boldsymbol{\beta}_{z} + \tilde{\mathbf{X}}_i^T \tilde{\mathbf{\beta}}_{1} \\
  &\quad + \exp\left(\eta_0 + \mathbf{Z}_i^T \boldsymbol{\eta}_z + \tilde{\mathbf{X}}_i^T \tilde{\mathbf{\eta}}_{1} \right) q_\tau
  \end{align*}

- Model:
  \begin{equation*}
  Y_i = \lambda_{0, \tau} + \mathbf{Z}_{i}^T \boldsymbol{\lambda}_{Z, \tau} + \tilde{\mathbf{X}}_i^T \tilde{\boldsymbol \lambda}_{1, \tau} + \epsilon_i
  \end{equation*}
<!-- - Approach suggested by literature: $\epsilon_i \sim \mathrm{MGAL}_\tau$. Valid, but probably inefficient.  -->
- $\epsilon_i \sim \mathrm{GAL}_\tau(0, \sigma_i, \gamma)$ where $\log \sigma_i = \kappa_0 + \mathbf{Z}_{i}^T \boldsymbol{\kappa}_Z + \tilde{\mathbf{X}}_i^T \tilde{\boldsymbol \kappa}_1$


### Simulations (Preliminary)

- Choose $D = 1$, $I = [0,1]$, $\delta_i = \mathrm{N}(0,1)$, $N = 100$
- Generate $\omega_i, \phi_i \sim U(0,1)$
- Set $X_i = \sin(2\pi(\omega_i t + \phi_i))$
- Generate $Z_i \sim N(0,1)$
- $\beta_0 = 0$, $\beta_z = 1$, $\beta_1(t) = \cos(2\pi t)$ 
- $\eta_0 = 0$, $\eta_z = -.1$, $\eta_1(t) =.1 \cos(2 \pi (t + 1/8))$ 

::: columns

:::: column
![Functional covariates](code_by_nico/functional/slides/xs.png)
::::

:::: column
![Histogram](code_by_nico/functional/slides/ys.png)
::::

:::

### Results

- Fit using brms
- Normal priors 
- 2000 iterations
- Divergent transitions

\begin{table}[!htbp] \centering 
  \caption{Results} 
  \label{} 
\begin{tabular}{@{\extracolsep{5pt}} ccccc} 
\\[-1.8ex]\hline 
\hline \\[-1.8ex] 
 & Estimate & Est.Error & l-95\% CI & u-95\% CI \\ 
\hline \\[-1.8ex] 
Intercept & $$-$0.875$ & $0.129$ & $$-$1.133$ & $$-$0.634$ \\ 
sigma\_Intercept & $$-$0.999$ & $0.151$ & $$-$1.326$ & $$-$0.728$ \\ 
Z & $1.049$ & $0.119$ & $0.805$ & $1.282$ \\ 
\hline \\[-1.8ex] 
\end{tabular} 
\end{table} 

### Diagnostics

::: columns

:::: column
![Pairs](code_by_nico/functional/slides/pairs.png)
::::

:::: column
![Trace](code_by_nico/functional/slides/trace.png)
::::

:::

- Challenging to get posterior predictive checks











<!-- - This case was already in Zoh (2023), even though it was not considered explicitely -->
<!-- 
# Project 2: Bayesian MIMIC

## Background

### Literature: MIMIC-ME

- Tekwe et al. (2014):
  \begin{eqnarray}
Y_{ij} &=& \beta_{0j} + \beta_{ys,j}S_i+ \epsilon_{y,ij}; \nonumber  \\ % \label{Model.2_eq1}  \\
S_i &=&  f_{x}(X_{i}) + \sum^{M}_{m=1} h_{z_m}(Z_m) + \eta_i; \label{Model.2_eq1}   \\ %  \label{Model.2_eq2} \\
W_{i,k} &=& X_{i} + U_{w,ik},\; \mbox{for}\; k = 1,\cdots, K\nonumber  \\
X_i | \mathbf{Z}_{i} &  = & \mu_l + \boldsymbol\alpha_{z}^T\mathbf{Z}_i + \epsilon_{x}\; \nonumber 
\end{eqnarray}

---

![](code_by_nico/functional/slides/tik.png)

## Bayesian estimation 

### First results

- Latent factors can be treated as missing data, but this is not straightforward to implement 
- Chains are very inefficient 

![Results](code_by_nico/functional/slides/sem.png)
 -->









