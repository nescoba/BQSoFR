library(brms)

N <- 3000
M <- 2
K <- 4
J <- 3

mu <- 0
al <- matrix(c(-1, 1), ncol = 1)
Z <- matrix(rnorm(M * N, sd = .5), ncol = N)
ep_x <- matrix(rnorm(N, sd = 0.1), ncol = N)
X <- t(al) %*% Z + ep_x

U <- matrix(rnorm(K * N, sd = .2), ncol = N)
W <- sweep(U, 2, X, "+")

h_1 <- function(x) -x
h_2 <- function(x) 3 * x
f <- function(x) 2 * x
eta <- rnorm(N)
S <- f(X) + h_1(Z[1, ]) + h_2(Z[2, ]) + eta
S <- matrix(S, ncol = N)

beta0 <- matrix(rep(0, J * N), ncol = N)
beta1 <- matrix(rep(1, J), ncol = 1)
ep_y <- matrix(rnorm(J * N, sd = 0.1), ncol = N)
Y <- beta0 + beta1 %*% S + ep_y

data <- as.data.frame(t(rbind(Z, X, W, Y)))
names(data) <- c("Z1", "Z2", "X", "W1", "W2", "W3", "W4", "Y1", "Y2", "Y3")

data$Xm <- as.numeric(NA)
data$Sm <- as.numeric(NA)


bf2 <- bf(Sm | mi() ~ mi(Xm) + Z1 + Z2)
bf3 <- bf(Y1 ~ 0 + mi(Sm))
bf4 <- bf(Y2 ~ 0 + mi(Sm))
bf5 <- bf(Y3 ~ 0 + mi(Sm))
bf8 <- bf(Xm | mi() ~ Z1 + Z2)
bf9 <- bf(W1 ~ 0 + mi(Xm))
bf10 <- bf(W2 ~ 0 + mi(Xm))
bf11 <- bf(W3 ~ 0 + mi(Xm))
bf12 <- bf(W4 ~ 0 + mi(Xm))

priors <- c(
    prior(normal(-1, 1), class = b, coef = Z1, resp = Sm),
    prior(normal(3, 1), class = b, coef = Z2, resp = Sm),
    prior(normal(2, 1), class = b, coef = miXm, resp = Sm),
    prior(normal(1, 1), class = b, coef = miSm, resp = Y1),
    prior(normal(1, 1), class = b, coef = miSm, resp = Y2),
    prior(normal(1, 1), class = b, coef = miSm, resp = Y3),
    prior(normal(-1, 1), class = b, coef = Z1, resp = Xm),
    prior(normal(1, 1), class = b, coef = Z2, resp = Xm),
    prior(normal(1, 0.0001), class = b, coef = miXm, resp = W1),
    prior(normal(1, 0.0001), class = b, coef = miXm, resp = W2),
    prior(normal(1, 0.0001), class = b, coef = miXm, resp = W3),
    prior(normal(1, 0.0001), class = b, coef = miXm, resp = W4)
)

model <- brm(bf2 + bf3 + bf4 + bf5 + bf8 + bf9 + bf10 + bf11 + bf12 + set_rescor(FALSE),
    data = data, prior = priors, iter = 20000, cores = 4
)

summary(model)$fixed

summary(model)$spec_pars
