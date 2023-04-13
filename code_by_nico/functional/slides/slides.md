---
title: Bayesian Methods for Measurement Error Problems 
author: Nicolas Escobar 
header-includes:
  - \usepackage{listings}
---

# Project 1: BQR for Scalar-on-Function Regression with Heteroskedastic Data 

## Background 

### General Introduction

- Quantile regression (QR) has gained popularity as a robust alternative to ordinary regression
- QR is formulated as an optimization problem using a loss function $\rho_\tau$, where $\tau$ is the desired quantile
- Bayesian Quantile Regression (BQR) offers the usual adventages of the Bayesian framework, like quantifying uncertainty
- BQR is based on a likelihood function called the Asymmetric Laplace (AL) distribution whose kernel includes $\rho_\tau$

### GAL Distribution

- AL turns out to be too restrictive 
- Yan and Kottas (2017) proposed the Generalized Asymmetric Laplace distribution (GAL)
- It can be reparametrized with: 
    - A location parameter $\mu$
    - A scale parameter $\sigma$ 
    - A shape parameter $\gamma$
- Closed forms for the pdf and cdf can be derived, but they are cumbersome

### Sampling from the posterior

- In Bayesian statistics, every parameter has a distribution
- We sample from the distribution of $\mu$, $\sigma$ and $\gamma$ using MCMC
- In recent years, Hamiltonian Monte Carlo (HMC) has gained popularity
- Implemented in the \lstinline!brms! library in R
- Uses the same formula syntax as \lstinline!lme4!
- Supports non-parametric methods through \lstinline!mgcv!


### Low dimensional ilustration of our problem

-  Consider data generated according to  
    \begin{equation*}
    y_i = x_i^2 + \epsilon_i
    \end{equation*}
    where $\epsilon_i \sim N(0, \exp(4x))$

![Heteroskedastic data](code_by_nico/functional/slides/htskdata.png){width=50%}

### Warning {#fragile .fragile}

- Quantile regression should be able to handle this kind of heteroskedasticity 
- Turns out you have to be careful:
    \begin{lstlisting}
    model1 <- brm(y ~ s(x), family = 'GAL')
    model2 <- brm(y ~ s(x), sigma ~ s(x), 
                    family = 'GAL')
    \end{lstlisting}

![With and Without $\sigma$](code_by_nico/functional/slides/wandwo.png){width=50%}



