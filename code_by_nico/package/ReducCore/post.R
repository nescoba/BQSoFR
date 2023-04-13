meth <- c("Wrep", "FUI", "Wnaiv")


FinalMeth <- function(meth, Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop) {
    switch(meth,
        Wrep = BQBayes.ME_SoFRFuncSimWrep(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop),
        FUI = BQBayes.ME_SoFRFuncSimFxFUI(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop),
        Wnaiv = BQBayes.ME_SoFRFuncSimFxNaiveW(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop)
    )
}



#-- Function choices to run these in parallel

FuncChoic <- function(Val, Y, W, M, Z, a, Nsim, sig02.del, tauval, Delt, alpx, tau0, X) {
    switch(Val,
        "sim" = BQBayes.ME_SoFRFuncSim(Y, W, M, Z, a, Nsim, sig02.del, tauval, Delt, alpx, tau0, X),
        "ind" = BQBayes.ME_SoFRFuncIndiv(Y, W, M, Z, a, Nsim, sig02.del, tauval, Delt, alpx, tau0, X)
    )
}


#-------------------------------------------------------------------
#-------------------------------------------------------------------
#--- Data simulation
#-------------------------------------------------------------------
#-------------------------------------------------------------------


#--- Basis function
B.basis <- function(x, knots) {
    delta <- knots[2] - knots[1]
    n <- length(x)
    K <- length(knots)
    B <- matrix(0, n, K + 1)
    for (jj in 1:(K - 1))
    {
        act.inds <- (1:n)[(x >= knots[jj]) & (x <= knots[jj + 1])]
        act.x <- x[act.inds]
        resc.x <- (act.x - knots[jj]) / (knots[jj + 1] - knots[jj])

        B[act.inds, jj] <- (1 / 2) * (1 - resc.x)^2
        B[act.inds, jj + 1] <- -(resc.x^2) + resc.x + 1 / 2
        B[act.inds, jj + 2] <- (resc.x^2) / 2
    }
    return(B)
}


ar1_cor <- function(n, rho) {
    exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) -
        (1:n - 1))
    rho^exponent
}
#--- Distributon of errors for epsilon


#--  Options are: distF = "Rst", "Norm","MixNorm", "Gam","Gam2"
DistEps <- function(n, distF, sig.e) {
    pi.g <- .5
    ei <- switch(distF,
        "Gam2" = rgamma(n = n, shape = 1, scale = 1.5) - 1.5,
        "Gam" = rgamma(n = n, shape = 1, scale = 1.5) - 1.5,
        "Rst" = sn::rst(n = n, xi = 0, omega = .5, nu = 5, alpha = 2),
        "Norm" = rnorm(n = n, sd = sig.e),
        # "MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
        "MixNorm" = rnormmix(n = n, lambda = rep(pi.g, 2), mu = c(-2.0, 2.0), sigma = (rep(.75 * sig.e, 2)))
        # Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
    )
    ei
}

#--- Distributon of errors for U (the measurement error)
# @options are : "Rmst" and "Mvnorm"
DistUw <- function(n, distF, Sig) {
    p0 <- nrow(Sig)
    p10 <- round(0.5 * p0)
    p11 <- p0 - p10

    ei <- switch(distF,
        "Rmst" = sn::rmst(n = n, xi = c(rep(-2, p10), rep(2, p11)), Omega = Sig, alpha = rep(2, p0), nu = 5),
        "Mvnorm" = mvtnorm::rmvnorm(n = n, mean = rep(0, nrow(Sig)), sigma = Sig)
        # "MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
        # "MixNorm" = rnormmix(n=n, lambda = rep(pi.g,2), mu = c(-2.0,2.0), sigma = (rep(sig.e,2)))
        # Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
    )
    scale(ei, center = T, scale = F)
}

#---- Cov matrix type
# @Met = CS, Ar1
CovMat <- function(m, Met, sig, rho) {
    ei <- switch(Met,
        "CS" = (ones(m, m) - diag(rep(1, m))) * ((sig^2) * rho) + diag(rep(sig^2, m)),
        "Ar1" = ar1_cor(m, rho) * (sig^2) # CVTuningCov::AR1
        # "MixNorm" = Prob0*rnorm(n=n, mean=-1.75, sd=.5*sig.e) + (1 - Prob0)*rnorm(n=n, mean=1.75*sig.e, sd=.5*sig.e)
        # "MixNorm" = rnormmix(n=n, lambda = rep(pi.g,2), mu = c(-2.0,2.0), sigma = (rep(sig.e,2)))
        # Prob0*rnorm(n=n, mean=-2., sd=.5) + (1 - Prob0)*rnorm(n=n, mean=2., sd=.5)
    )
    ei
}


#---  Beta function
Betafunc <- function(t, met) {
    switch(met,
        f1 = 1. * sin(2 * pi * t) + 1.5,
        f2 = 1 / (1 + exp(4 * 2 * (t - .5))),
        f3 = 1. * sin(pi * (8 * (t - .5)) / 2) / (1 + (2 * (8 * (t - .5))^2) * (sign(8 * (t - .5)) + 1)) + 1.5,
        f4 = sin(pi * (16 * (t - .5)) / 2) / (1 + (2 * (16 * (t - .5))^2) * (.5 * sin(8 * (t - .5)) + 1)),
        f5 = sin(pi * (16 * (t - .5)) / 2) / (1 + (2 * (16 * (t - .5))^2) * (sin(8 * (t - .5)) + 1))
    )
}

#-- Simulate data with constant variance
DataSim <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep = c("Rst", "Norm", "MixNorm"), distFUe = c("Rmst", "Mvnorm"), distFepWe = c("Rmst", "Mvnorm"), CovMet = c("CS", "Ar1"), sig.e, idf = c(1:4)) {
    a <- seq(0, 1, length.out = t) # time points at which obs. are takes
    met <- c("f1", "f2", "f3", "f4", "f5")
    Delt <- Deltafun(a, scale = 1., loc = .5)

    # plotting
    par(mfrow = c(2, 2))
    plot(a, Betafunc(a, met[idf]), type = "l")
    plot(a, Delt, type = "l")

    #---------------------------------------------------------------------
    #---- Simulate X
    #---------------------------------------------------------------------
    SigX <- CovMat(t, CovMet, sig.x, rhox) # ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2)
    MeanX <- Betafunc(a, met[idf]) # sinpi(2*a)
    MeanX2 <- Betafunc(a, met[idf]) # sinpi(2*a)
    X_t <- matrix(MeanX, ncol = t, nrow = n, byrow = T) + DistUw(n, distFUe, SigX) # rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
    plot(a, MeanX, type = "l", lwd = 3)

    # We generate the basis functions from bsplines
    a <- seq(0, 1, length.out = t)
    bs2 <- bs(a, df = pn, intercept = T)
    bs20 <- bs(a, df = pn, intercept = T, degree = 2)
    bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
    # bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))

    # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1))
    # if(distF == "Norm"){
    pi.g <- .5
    Prob0 <- rbinom(n = n, size = 1, prob = pi.g)

    Z <- matrix(rnorm(n = n * j0, sd = 1), ncol = j0)
    Z <- scale(Z)
    # Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z);

    Betaz <- rnorm(n = j0 + 1)
    Betaz <- c(-1.05, .57) * c(1, .35, .35)[idf] # c(0., -1.05, .57) # c(1.87, -1.05, .57)
    Zmod <- Z # cbind(1, Z)
    Val <- 1 #<- caseN %in%paste("case",1:4,"a",sep="")
    #* (1-Val)
    ei <- DistEps(n, distFep, sig.e) * c(1, 1, .5, 1)[idf]
    BetatF <- (Betafunc(a, met[idf]) - mean(Betafunc(a, met[idf]))) * c(1, 2, 2)[idf]
    fx0 <- crossprod(t(X_t), BetatF) / length(a)
    Y <- Zmod %*% Betaz * Val + fx0 + ei * fx0 #* (crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))
    # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix


    # f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
    par(mfrow = c(2, 2))
    MASS::truehist(Y)
    MASS::truehist(ei)
    MASS::truehist(c(Zmod %*% Betaz))
    MASS::truehist(fx0)

    # print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
    print(mean(Zmod %*% Betaz * Val + fx0) / sd((ei - mean(ei))))


    # Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
    #---------------------------------------------------------------------
    # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
    #---------------------------------------------------------------------
    SigU <- CovMat(t, CovMet, sig.u, rho.u) # ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2)
    MeanU <- numeric(t)
    U <- DistUw(n, distFUe, SigU) # mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
    W <- X_t + U # (n,t) W matrix of surrogate values
    Wn <- (W %*% bs2) / length(a)

    #---------------------------------------------------------------------
    # We generate the IV with errors from the Normal distribution (Equation (3))
    #---------------------------------------------------------------------
    SigW <- ones(t, t) * rho.m * sig.w * sig.w
    diag(SigW) <- (sig.w^2)
    MeanW <- numeric(t)
    omega <- DistUw(n, distFUe, SigW) # mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)

    M <- X_t %*% diag(Delt) + omega # (n,t) M matrix
    # Mn <- (M%*%bs2)/length(a)
    # Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
    # plot(a, Delthat, type="l")
    Mn_Mod <- M %*% diag(1 / Delt)
    # bs2 <- bs(a, df = pn, intercept = T)
    Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
    Wn <- (W %*% bs2) / length(a)

    Data <- list(
        a = a, bs2 = bs2, Z = Z, Y = Y, X = X_t, DataN = list(Mn = Mn, Wn = Wn, Xn = (X_t %*% bs2) / length(a)),
        Data = list(M = M, W = W, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF
    )
    Data
}


#-- Simulate data with constant variance
DataSimHeter <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep = c("Rst", "Norm", "MixNorm"), distFUe = c("Rmst", "Mvnorm"), distFepWe = c("Rmst", "Mvnorm"), CovMet = c("CS", "Ar1"), sig.e, idf = c(1:4)) {
    # Nbasis0 =  c(5, 6, 15, 15) # pn = [which(n == c(100, 200,500,1000))]
    # J = K = k  = pn = pn00 =  Nbasis0[which(c(100,200,500, 1000) == n)]
    a <- seq(0, 1, length.out = t) ## time points at which obs. are takes

    met <- c("f1", "f2", "f3", "f4", "f5")




    P.mat <- function(K) {
        # penalty matrix
        D <- diag(rep(1, K))
        D <- diff(diff(D))
        P <- t(D) %*% D
        return(P)
    }

    # Deltafun <- function(a, scale, loc){ (.25 - (a - .5)^2)*scale + rep(loc,length(a))}
    Deltafun <- function(a, scale, loc) {
        (1.25 + Betafunc(a, met[idf])) * scale + rep(loc, length(a))
    }

    Delt <- Deltafun(a, scale = 1., loc = .5)
    # P.mat <- function(K){
    #   return(ppls::Penalty.matrix(K,order=2))
    # }
    par(mfrow = c(2, 2))
    plot(a, Betafunc(a, met[idf]), type = "l")
    plot(a, Delt, type = "l")

    # P.mat(3)

    #---------------------------------------------------------------------
    #---- Simulate X
    #---------------------------------------------------------------------
    # sig.x = .5
    SigX <- CovMat(t, CovMet, sig.x, rhox) # ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2)
    MeanX <- Betafunc(a, met[idf]) # sinpi(2*a)
    MeanX2 <- Betafunc(a, met[idf]) # sinpi(2*a)
    X_t <- matrix(MeanX, ncol = t, nrow = n, byrow = T) + DistUw(distFUe, SigX) # rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
    plot(a, MeanX, type = "l", lwd = 3)

    # We generate the basis functions from bsplines
    a <- seq(0, 1, length.out = t)
    bs2 <- bs(a, df = pn, intercept = T)
    bs20 <- bs(a, df = pn, intercept = T, degree = 2)
    bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
    # bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))

    # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1))
    # if(distF == "Norm"){
    pi.g <- .5
    Prob0 <- rbinom(n = n, size = 1, prob = pi.g)

    Z <- matrix(rnorm(n = n * j0, sd = 1), ncol = j0, )
    Z <- scale(Z)
    # Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z);

    Betaz <- rnorm(n = j0 + 1)
    Betaz <- c(-1.05, .57) * .35 # c(0., -1.05, .57) # c(1.87, -1.05, .57)
    Zmod <- Z # cbind(1, Z)
    Val <- 1 #<- caseN %in%paste("case",1:4,"a",sep="")
    #* (1-Val)
    ei <- DistEps(n, distFep, sig.e)
    BetatF <- (Betafunc(a, met[idf]) - mean(Betafunc(a, met[idf]))) * 2
    fx0 <- crossprod(t(X_t), BetatF) / length(a)
    Ei <- ei * fx0
    Y <- Zmod %*% Betaz * Val + fx0 + .5 * Ei #* (crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))
    # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix


    # f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
    par(mfrow = c(2, 2))
    MASS::truehist(Y)
    MASS::truehist(.5 * ei * fx0)
    MASS::truehist(c(Zmod %*% Betaz))
    MASS::truehist(fx0)

    # print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
    print(mean(Zmod %*% Betaz * Val + fx0) / sd((ei - mean(ei))))


    # Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
    #---------------------------------------------------------------------
    # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
    #---------------------------------------------------------------------
    SigU <- CovMat(t, CovMet, sig.u, rho.u) # ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2)
    MeanU <- numeric(t)
    U <- DistUw(distFUe, SigU) # mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
    W <- X_t + U # (n,t) W matrix of surrogate values
    Wn <- (W %*% bs2) / length(a)

    #---------------------------------------------------------------------
    # We generate the IV with errors from the Normal distribution (Equation (3))
    #---------------------------------------------------------------------
    SigW <- ones(t, t) * rho.m * sig.w * sig.w
    diag(SigW) <- (sig.w^2)
    MeanW <- numeric(t)
    omega <- DistUw(distFUe, SigU) # mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)

    M <- X_t %*% diag(Delt) + omega # (n,t) M matrix
    # Mn <- (M%*%bs2)/length(a)
    # Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
    # plot(a, Delthat, type="l")
    Mn_Mod <- M %*% diag(1 / Delt)
    # bs2 <- bs(a, df = pn, intercept = T)
    Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
    Wn <- (W %*% bs2) / length(a)

    Data <- list(a = a, bs2 = bs2, Z = Z, Y = Y, X = X_t, DataN = list(Mn = Mn, Wn = Wn, Xn = (X_t %*% bs2) / length(a)), Data = list(M = M, W = W, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF)
    Data
}


#--- Dataset with replicated obs
#-- Simulate data with constant variance

# Z <- matrix(rnorm(n=n*(j0-1), sd = 1), ncol = c(j0-1)); Z <- scale(Z);
# Z <- cbind(Z, rbinom(n = n, prob=pi.g, size=1))

DataSimReplicate <- function(pn, n, t, a, j0 = 2, rhox, sig.x, rho.u, sig.u, rho.m, sig.w, distFep = c("Rst", "Norm", "MixNorm"), distFUe = c("Rmst", "Mvnorm"), distFepWe = c("Rmst", "Mvnorm"), CovMet = c("CS", "Ar1"), sig.e, idf = c(1:4), nrep, Wdist, Z) {
    # @parms Wdist = "norm" or "count"
    # Nbasis0 =  c(5, 6, 15, 15) # pn = [which(n == c(100, 200,500,1000))]
    # J = K = k  = pn = pn00 =  Nbasis0[which(c(100,200,500, 1000) == n)]
    a <- seq(0, 1, length.out = t) ## time points at which obs. are takes

    met <- c("f1", "f2", "f3", "f4", "f5")

    B.basis <- function(x, knots) {
        delta <- knots[2] - knots[1]
        n <- length(x)
        K <- length(knots)
        B <- matrix(0, n, K + 1)
        for (jj in 1:(K - 1))
        {
            act.inds <- (1:n)[(x >= knots[jj]) & (x <= knots[jj + 1])]
            act.x <- x[act.inds]
            resc.x <- (act.x - knots[jj]) / (knots[jj + 1] - knots[jj])

            B[act.inds, jj] <- (1 / 2) * (1 - resc.x)^2
            B[act.inds, jj + 1] <- -(resc.x^2) + resc.x + 1 / 2
            B[act.inds, jj + 2] <- (resc.x^2) / 2
        }
        return(B)
    }


    P.mat <- function(K) {
        # penalty matrix
        D <- diag(rep(1, K))
        D <- diff(diff(D))
        P <- t(D) %*% D
        return(P)
    }

    # Deltafun <- function(a, scale, loc){ (.25 - (a - .5)^2)*scale + rep(loc,length(a))}
    Deltafun <- function(a, scale, loc) {
        (1.25 + Betafunc(a, met[idf])) * scale + rep(loc, length(a))
    }

    Delt <- Deltafun(a, scale = 1., loc = .5)
    # P.mat <- function(K){
    #   return(ppls::Penalty.matrix(K,order=2))
    # }
    par(mfrow = c(2, 2))
    plot(a, Betafunc(a, met[idf]), type = "l")
    plot(a, Delt, type = "l")

    # P.mat(3)

    #---------------------------------------------------------------------
    #---- Simulate X
    #---------------------------------------------------------------------
    # sig.x = .5
    SigX <- CovMat(t, CovMet, sig.x, rhox) # ones(t,t)*rhox*sig.x*sig.x; diag(SigX) <- (sig.x^2)
    MeanX <- Betafunc(a, met[1]) # Betafunc(a, met[idf]) #sinpi(2*a)
    MeanX2 <- Betafunc(a, met[1]) # Betafunc(a, met[idf]) #sinpi(2*a)
    X_t <- matrix(MeanX, ncol = t, nrow = n, byrow = T) + DistUw(n, distFUe, SigX) # rbind(mvtnorm::rmvnorm(n=n*.5,mean = MeanX, sigma = SigX), mvtnorm::rmvnorm(n=n*.5,mean = MeanX2, sigma = SigX))
    plot(a, MeanX, type = "l", lwd = 3)

    # We generate the basis functions from bsplines
    a <- seq(0, 1, length.out = t)
    bs2 <- bs(a, df = pn, intercept = T)
    bs20 <- bs(a, df = pn, intercept = T, degree = 2)
    bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
    # bs2 <- B.basis(a, knots= seq(0,1, length.out = pn-1))

    # We generate Yi the scalar response with error terms from Normal (0,.05) (Equation (1))
    # if(distF == "Norm"){
    pi.g <- .65
    Prob0 <- rbinom(n = n, size = 1, prob = pi.g)

    # Z <- matrix(rnorm(n=n*(j0-1), sd = 1), ncol = c(j0-1)); Z <- scale(Z);
    # Z <- cbind(Z, rbinom(n = n, prob=pi.g, size=1))
    # Z[,2] <- c(mapply(rnorm, n=1, mean = crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))); Z <- scale(Z);

    Betaz <- rnorm(n = j0 + 1)
    Betaz <- c(-1.05, .57) * c(1, .35, .35)[idf] # c(0., -1.05, .57) # c(1.87, -1.05, .57)
    Betaz <- c(-1.65, 0.91, -0.84) # rnorm(n = j0) + .2; #c(0., -1.05, .57) # c(1.87, -1.05, .57)
    Zmod <- Z # cbind(1, Z)
    Val <- 1 #<- caseN %in%paste("case",1:4,"a",sep="")
    #* (1-Val)
    if (distFep != "Gam") {
        ei <- DistEps(n, distFep, sig.e) * c(1, 1, .5, 1)[idf] * 3
        BetatF <- (Betafunc(a, met[idf]) - mean(Betafunc(a, met[idf]))) * c(1, 2, 2, 2)[idf]
        fx0 <- crossprod(t(X_t), BetatF) / length(a)
    } else {
        ei <- DistEps(n, distFep, sig.e) * 1
        BetatF <- 1.5 * (Betafunc(a, met[idf]) - mean(Betafunc(a, met[idf]))) * c(1, 2, 2, 2)[idf]
        fx0 <- crossprod(t(X_t), BetatF) / length(a)
    }

    Y <- c(Zmod %*% Betaz * Val) + c(fx0) + (ei - mean(ei)) * c(fx0) #* (crossprod(t(X_t),Betafunc(a,met[idf]))/length(a)) #(ei-mean(ei))
    # #d0*rnorm(n, sd = sig.e) + (1-d0)*(rgamma(n=n,shape=1,scale=1.5) - 1.5) # (n,1) Y matrix


    # f0 <- function(t) { X_t[1,]*Betafunc(t,met[idf])}
    par(mfrow = c(2, 2))
    MASS::truehist(Y)
    MASS::truehist(ei)
    MASS::truehist(c(Zmod %*% Betaz))
    MASS::truehist(fx0)

    # print(mean(Zmod%*%Betaz*Val + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a))/sd((ei-mean(ei))))
    print(mean(Zmod %*% Betaz * Val + fx0) / sd((ei - mean(ei))))


    # Y <- 0 + crossprod(t(X_t),Betafunc(a,met[idf]))/length(a) +  rgamma(n=n,shape=1,scale=1.5) - 1.5 # (n,1) Y matrix
    #---------------------------------------------------------------------
    # We generate the observed surrogates values W with errors from the Skew Normal distribution (Equation (2))
    #---------------------------------------------------------------------
    SigW <- ones(t, t) * rho.m * sig.w * sig.w
    diag(SigW) <- (sig.w^2)
    MeanW <- numeric(t)
    SigU <- CovMat(t, CovMet, sig.u, rho.u) # ones(t,t)*rho.u*sig.u*sig.u; diag(SigU) <- (sig.u^2)
    MeanU <- numeric(t)
    W_array <- array(NA, c(n, t, nrep))
    M_array <- array(NA, c(n, t, nrep))

    if (Wdist == "norm") {
        for (li in 1:nrep) {
            U <- DistUw(n, distFUe, SigU) # mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
            W_array[, , li] <- X_t + U # (n,t) W matrix of surrogate values

            omega <- DistUw(n, distFUe, SigW) # mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
            M_array[, , li] <- X_t %*% diag(Delt) + omega # (n,t) M matrix
            # Me <- DistUw(n,distFUe, Sig)#mvtnorm::rmvnorm(n=n,mean = MeanU, sigma = SigU)
            # M_array[,,li] = X_t + U  # (n,t) W matrix of surrogate values
        }
    } else {
        temp <- array((X_t), dim = c(n, t, m_w))
        W_array <- apply(temp, c(1, 2, 3), function(s) {
            rpois(1, exp(s))
        })
    }

    Wn <- (W_array[, , 1] %*% bs2) / length(a)

    #---------------------------------------------------------------------
    # We generate the IV with errors from the Normal distribution (Equation (3))
    #---------------------------------------------------------------------
    # SigW <- ones(t,t)*rho.m*sig.w*sig.w; diag(SigW) <- (sig.w^2)
    # MeanW <- numeric(t)
    # omega <- DistUw(n,distFUe, SigW) #mvtnorm::rmvnorm(n=n,mean = MeanW, sigma = SigW)
    #
    # M <- X_t%*%diag(Delt) + omega  # (n,t) M matrix
    # # Mn <- (M%*%bs2)/length(a)
    # #Delthat <- rep(1, length(Delt))#Delt # lowess(a, (colMeans(M)/colMeans(W)), f=1/6)$y # c(abs(colMeans(M)/(colMeans(W)))) #   lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
    # #plot(a, Delthat, type="l")
    # Mn_Mod <- M%*%diag(1/Delt)
    # #bs2 <- bs(a, df = pn, intercept = T)
    # Mn <- (Mn_Mod%*%bs2)/length(a) #(M%*%bs2)/length(a)
    # Wn <- (W%*%bs2)/length(a)
    Xn <- (X_t %*% bs2) / length(a)

    Wrep <- array(NA, c(nrep, t, n))
    for (i in 1:n) {
        Wrep[, , i] <- t(W_array[i, , ])
    }

    Data <- list(
        a = a, bs2 = bs2, Z = Z, Y = Y, X = X_t, Xn = Xn,
        Data = list(W = W_array, Wrep = Wrep, M = W_array[, , 1], Mall = M_array, X = X_t), Delt = Delt, BetaZ = Betaz, Betat = BetatF
    )
    Data
}


#--- FUI appaproach

FUI <- function(model = "gaussian", smooth = FALSE, data, silent = FALSE) {
    ##' @param model is the model for apporixation
    ##   model = gaussian, poisson
    ##' @param data is the observed measurement of X(t) with repeated measures, which is in 3 dimension [n, t, m_w]
    ##' @param smooth whether conduct the smoothing step or not (Step II)
    ##' @param silent whether to show descriptions of each step

    ## The output of this function is the approximation of the true function covariate X(t),
    ##    where each column represent each time point and each row represent each subject



    library(lme4) ## mixed models
    # library(refund) ## fpca.face
    library(dplyr) ## organize lapply results
    # library(progress) ## display progress bar
    library(mgcv) ## smoothing in step 2
    # library(mvtnorm) ## joint CI
    # library(parallel) ## mcapply

    ### create a dataframe for analysis
    m_w <- dim(data)[3] ## number of repeated measures
    t <- dim(data)[2] ## number of observations on the functional domain (time points)
    n <- dim(data)[1] ## number of subjects
    M <- NULL
    i <- 1
    while (i <= m_w) {
        M <- rbind(M, data[, , i])
        i <- i + 1
    } ### convert the 3d array to 2d matrix

    data.M <- data.frame(M = I(M), seqn = rep(1:n, m_w))

    ##########################################################################################
    ## Step 1 (Massive Univariate analysis)
    ##########################################################################################
    if (silent == FALSE) print("Step 1: Massive Univariate Mixed Models")

    if (model == "gaussian") {
        fit <- data.M %>%
            dplyr::select(M) %>%
            apply(2, function(s) {
                temp <- data.frame(M = s, seqn = data.M$seqn)
                mod <- suppressMessages(lmer(M ~ (1 | seqn), data = temp))
                return(predict(mod))
            })
    } else {
        fit <- data.M %>%
            dplyr::select(M) %>%
            apply(2, function(s) {
                temp <- data.frame(M = s, seqn = data.M$seqn)
                mod <- suppressMessages(glmer(M ~ (1 | seqn), data = temp, family = model))
                return(predict(mod))
            })
    }

    approx <- fit[!duplicated(data.M$seqn), ] ## each column represents one time point

    ##########################################################################################
    ## Step 2 (Smoothing)
    ##########################################################################################

    if (smooth == TRUE) {
        if (silent == FALSE) print("Step 2: Smoothing")
        nknots <- min(round(t / 4), 35) ## number of knots for penalized splines smoothing
        argvals <- seq(0, 1, length.out = t) ## locations of observations on the functional domain
        approx <- t(apply(approx, 1, function(x) gam(x ~ s(argvals, bs = "cr", k = (nknots + 1)), method = "REML")$fitted.values))
    }

    return(approx)
}



### Fast short multivariate interference(FSMI) function ###

FSMI <- function(model = "gaussian", cov.model = "us", data) {
    ## @t is the total time points
    ## @model is the model for apporixation
    ##   model = gaussian, poisson
    ## @cov.model is the covaraince structure for random effect
    ##   cov.model = us, id, cs
    ## @data is the observed measurement of X(t) with repeated measures, in 3 dimension [n,t,m_w]
    ## The output of this function is the approximation of the true function covariate X(t),
    ##    where each column represent each time point and each row represent each subject

    m_w <- dim(data)[3] ## number of repeated measures
    t <- dim(data)[2] ## number of observations on the functional domain (time points)
    n <- dim(data)[1] ## number of subjects
    data.M <- matrix(data, prod(dim(data)[1:3])) %>% as.data.frame() ### convert the 3d array to 2d matrix
    data.M$seqn <- rep(1:n, prod(t, m_w)) ## when convert into matrix, each column comes after the first colum and then the third demension
    data.M$time <- rep(sapply(1:t, function(s) rep(s, n)), m_w)
    data.M$day <- lapply(1:m_w, function(s) rep(s, n * t)) %>% unlist()
    colnames(data.M)[1] <- "M"

    ind.time <- 1:t ### number of time point
    fit <- lapply(
        ind.time,
        function(s) {
            # print(s)

            ### select the interval of time for analysis
            if ((s - 1) < 1) {
                ind.interval <- 1:3
            } else if ((s + 1) > t) {
                ind.interval <- (t - 2):t
            } else {
                ind.interval <- (s - 1):(s + 1)
            }

            temp <- data.M %>%
                filter(time %in% ind.interval) %>% # only select the data of the selected time points
                mutate(time.cat = as.character(time))

            if (model == "gaussian") {
                if (cov.model == "us") { ## unstructured
                    mod <- suppressMessages(lmer(M ~ (1 | seqn) + (0 + time.cat | seqn), data = temp)) ## default setting for lmer is unstructured
                } else if (cov.model == "id") { ## independent
                    mod <- suppressMessages(lmer(M ~ (1 | seqn) + (1 | time.cat), data = temp))
                } # https://stats.stackexchange.com/questions/86958/variance-covariance-structure-for-random-effects-in-lme4
                # https://stats.stackexchange.com/questions/49775/variance-covariance-matrix-in-lmer
                else if (cov.model == "cs") { ## compound symmetric
                    mod <- suppressMessages(lme(M ~ 1,
                        random = list(seqn = ~1, seqn = pdCompSymm(~ time.cat - 1)),
                        data = temp
                    ))
                } else {
                    stop("wrong covariance matrix of random effect")
                }
            } else {
                if (cov.model == "us") { ## unstructured
                    mod <- suppressMessages(glmer(M ~ (1 | seqn) + (0 + time.cat | seqn), data = temp, family = model)) ## default setting for lmer is unstructured
                } else if (cov.model == "id") { ## independent
                    mod <- suppressMessages(glmer(M ~ (1 | seqn) + (1 | time.cat), data = temp, family = model))
                } # https://stats.stackexchange.com/questions/86958/variance-covariance-structure-for-random-effects-in-lme4
                # https://stats.stackexchange.com/questions/49775/variance-covariance-matrix-in-lmer
                else if (cov.model == "cs") { ## compound symmetric
                    mod <- suppressMessages(glme(M ~ 1,
                        random = list(seqn = ~1, seqn = pdCompSymm(~ time.cat - 1)),
                        data = temp, family = model
                    ))
                } else {
                    stop("wrong covariance matrix of random effect")
                }
            }
            # # Different variances
            # ## seperate variance per time level
            # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdDiag(~ time.cat-1)),data = temp)
            #
            # ## unstructured at time level
            # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdSymm(~ time.cat-1)),data = temp)
            #
            #
            # ## Compound symmetric at time level
            # mod =  lme(M ~time.cat, random = list(seqn= ~1, seqn =pdCompSymm(~ time.cat-1)),
            #            data = temp)
            # ## https://rpubs.com/samuelkn/CovarianceStructuresInR
            #

            # getVarCov(mod, type="marginal") ## get covariance matrix of y
            # VarCorr(mod, type="marginal") ## get the G matrix

            ## us = unstructured, id = independent, cs = compound symmetric, ar1 = autoregressive (1)

            ## us = unstructured, id = independent, cs = compound symmetric, ar1 = autoregressive (1)
            pred <- predict(mod, newdata = data.frame(time.cat = as.character(s), seqn = unique(temp$seqn)))
            return(pred)
        }
    ) ### end of lapply()


    approx <- matrix(unlist(fit), nrow = n, ncol = t) ## each column represents one time point

    return(approx)
}




#--- Find prior parameters for the inversge Gamma prior

SolParm <- function(parm) {
    val <- c(.2, 4)
    (val[1] - nimble::qinvgamma(0.0001, shape = parm[1], scale = parm[2]))^2 + (val[2] - nimble::qinvgamma(0.9999, shape = parm[1], scale = parm[2]))^2
}

#--------------------------------------------------------
#--- Other aproaches
#--------------------------------------------------------

library(stats)
library(sn)
library(splines)
library(ald)
library(mvtnorm)
library(boot)
library(fda)
# library(ggplot2)
library(lattice)
library(readxl)
library(plyr)
library(magrittr)
library(Matrix)
library(SparseM)
library(quantreg)
# library(corpcor)
library(MASS)
library(reshape)
library(grid)
library(gridExtra)
# library(knitr)
# library(doParallel)
# library(foreach)


### extrapolation function
extra_quad <- function(k, beta_hat_lambda, lambda, lambda_app) {
    ## k is the number of basis functions for the functional variable
    ## beta_hat_lambda is the estimated beta(t) from SIMEX simulation step
    ## lambda is the squence of lambda
    ## lambda_app = -1 for SIMEX estimators
    gamma <- rep(0, k)
    lambda2 <- lambda * lambda
    for (m in 1:k)
    {
        testr <- beta_hat_lambda[m, ]
        lr <- lm(testr ~ lambda + lambda2)
        coeff <- lr$coefficients
        gamma[m] <- coeff[1] + lambda_app * coeff[2] + lambda_app * lambda_app * coeff[3]
    }

    return(gamma)
}

#### functions used to generate data
met <- c("f1", "f2", "f3", "f4", "f5", "f6", "f7")
Betafuncoo <- function(t, met) {
    switch(met,
        f1 = sin(2 * pi * t), ## f1 = sin(2*pi*t),
        f2 = 1 / (1 + exp(4 * 2 * (t - .5))),
        f3 = sin(pi * (8 * (t - .5)) / 2) / (1 + (2 * (8 * (t - .5))^2) * (sign(8 * (t - .5)) + 1)),
        f4 = sin(pi * (16 * (t - .5)) / 2) / (1 + (2 * (16 * (t - .5))^2) * (.5 * sin(8 * (t - .5)) + 1)),
        f5 = sin(pi * (16 * (t - .5)) / 2) / (1 + (2 * (16 * (t - .5))^2) * (sin(8 * (t - .5)) + 1)),
        f6 = 1 / (1 + exp((t))),
        f7 = 1 / (1 + exp(4 * 2 * (t - .5))) + 1
    )
}


######## function for selecting number of basis ###
### based on BIC
### the output of this function is a BIC value
L_BIC <- function(k, Y, W, tau, EF = EF, time_interval) {
    ### Y is response, W is observed matrix of W(t);
    ### k is the number of basis and should be at least 4;
    ##  tau is quantile level;
    ## time_interval is interval of observed time
    cubic_bs <- bs(time_interval, df = k, degree = 3, intercept = TRUE)
    pred <- t(t(W) - colMeans(W)) %*% cubic_bs / length(time_interval)

    coef_quant <- rq(Y ~ pred + EF, tau = tau)$coefficients
    n <- length(Y)
    eps <- Y - cbind(rep(1, n), pred, EF) %*% coef_quant
    loss <- eps * ifelse(eps < 0, tau - 1, tau)

    re <- log(mean(loss)) + (k + 1) * log(n) / n
    return(re)
}



# main.delta=function(Y, W, M, Z, n,c,sd_x,sd_w,p_sim,fi,fx,seeds,sim_liter, deltat){
main.delta <- function(Y_norm, W_t, M_t, X_t, Z, delta_t, a, seeds = 123, sim_liter = 1, p_sim = 0.5) {
    require(fda)
    ## @seeds is the the seeding basis
    ## @n is the size of simulated data
    ## @c is the tuning parameter for generating delta(t) (delta(t) = c*sin(2*pi*t) +1)
    ## @sd_x is the sd of X(t)
    ## @sd_w is the sd of W(t)
    ## @p_sim is the quantile number (0.25, 0.5, 0.75, 0.95)
    ## @fi is the function for generating Y
    ## @fx is the function for generating X(t)
    ## @sim_lier is the number of replications we do for one simulation

    set.seed(seeds)
    t <- ncol(W_t) ## Number of time points
    k_max <- 7 ## The maximum number of basis functions to be considered
    # a  = seq(0, 1, length.out=t) #create a sequence between 0,1 of length = t
    n <- length(Y)

    # sd_m=0.25 ## standard deviation of instrumental variable
    # sigma_y = .1 ###standard deviation of Y
    # sigma_ef = 0.5 ###standard deviation of EF covariate
    # p = 0.6 ## prob for the EF binary predictor

    ############ create covariance matrix for error terms #################
    # rho_x=0.5 ## correlation coefficient of X(t)
    # rho_w=0.5 ## correlation coefficient of W(t)
    # rho_m=0.5 ## correlation coefficient of M(t)
    #
    # ###compound symmetric covariance matrix ###
    # Sigma_m=sd_m^2*(matrix(rho_m,nrow = t, ncol = t)+diag(rep(1-rho_m,t)))
    # Sigma_x=sd_x^2*(matrix(rho_x,nrow = t, ncol = t)+diag(rep(1-rho_x,t)))
    # Sigma_w=sd_w^2*(matrix(rho_w,nrow = t, ncol = t)+diag(rep(1-rho_w,t)))
    #
    zero <- rep(0, times = t)

    ############### create some matrices to store results #############

    #### functional variable #####
    beta_X <- matrix(nrow = t, ncol = sim_liter) # results matrix from the quantile regression based on the true X
    beta_simex <- matrix(nrow = t, ncol = sim_liter) # SIMEX estimator when assuming delta(t) is 1
    beta_naive <- matrix(nrow = t, ncol = sim_liter) ## naive estimator
    beta_simex.delta <- matrix(nrow = t, ncol = sim_liter) ## SIMEX estimator when delta(t) is estimated as raw ratio
    beta_simex.delta.known <- matrix(nrow = t, ncol = sim_liter) ## SIMEX estimator when delta(t) is known
    beta_simex.deltas <- matrix(nrow = t, ncol = sim_liter) ## SIMEX estimator when delta(t) is estimated as smoothed ratio

    ###### continuous error-free covariate ####
    beta_X_EF <- matrix(nrow = 1, ncol = sim_liter) ## ## benchmark estimator
    beta_naive_EF <- matrix(nrow = 1, ncol = sim_liter) # SIMEX estimator when assuming delta(t) is 1
    beta_W_EF <- matrix(nrow = 1, ncol = sim_liter) ### EF estimator when ignore ME in functional varibale
    beta_naive_EF.delta <- matrix(nrow = 1, ncol = sim_liter)
    beta_naive_EF.deltas <- matrix(nrow = 1, ncol = sim_liter)
    beta_naive_EF.delta.known <- matrix(nrow = 1, ncol = sim_liter)
    ###### binary error-free covariate ####
    beta_X_EF.b <- matrix(nrow = 1, ncol = sim_liter) ## benchmark estimator
    beta_naive_EF.b <- matrix(nrow = 1, ncol = sim_liter) ## SIMEX estimator when assuming delta(t) is 1
    beta_W_EF.b <- matrix(nrow = 1, ncol = sim_liter) ### EF estimator when ignore ME in functional variable
    beta_naive_EF.delta.b <- matrix(nrow = 1, ncol = sim_liter)
    beta_naive_EF.deltas.b <- matrix(nrow = 1, ncol = sim_liter)
    beta_naive_EF.delta.known.b <- matrix(nrow = 1, ncol = sim_liter)


    #### number of basis functions k ####
    selected_k_naive <- c() ## k based on BIC value of naive model
    selected_k_bench <- c() ## k based on BIC value of benchmark model (with X(t))
    selected_k_n <- c() ## k based on sample size

    #######  estimated delta(t) #####
    delta_t.hat <- matrix(nrow = t, ncol = sim_liter) ## raw delta(t)
    delta_t.hat.smooth <- matrix(nrow = t, ncol = sim_liter) ## smoothed delta(t)



    for (iter in 1:sim_liter) {
        #-------------------------------------------------------------#
        ################### Generate X, W, M, Y,and EF ################
        #-------------------------------------------------------------#

        ######### generate error terms #####
        # err_x = mvrnorm(n, zero, Sigma_x) ### error term of X(t)
        # err_w = mvrnorm(n, zero, Sigma_w)### error term of W(t)
        # err_m = mvrnorm(n, zero, Sigma_m)### error term of M(t)
        # err_ef = rnorm(n, 0, sigma_ef) ### error term of continuous error-free covariate
        #

        ### generate delta(t)
        # delta_t =c*Betafunc(a,"f1") +1 # c is the constant might change for different delta(t)
        #
        # X_t = matrix(rep(Betafunc(a,fx),n),nrow=n,byrow=TRUE) + err_x
        # M_t = t(apply(X_t, 1, function(s) delta_t*s)) + err_m
        # W_t = X_t + err_w
        #
        # EF = rnorm(n, 0, sigma_ef)  ## EF predictor that is NOT correlated with X(t)
        # EF.b = rbinom(n, 1,p) ## binary predictor
        #
        # #### generate Y ###
        # gamma.1 = 2 ## true association of Y and EF
        # gamma.3 = 0.6 ## true association of Y and biarny EF
        #
        # Y_norm = crossprod(t(X_t),Betafunc(a,fi))/length(a)+ ## true association of Y and x(t)
        #   crossprod(t(EF), gamma.1)+ ## true association of Y and EF
        #   crossprod(t(EF.b), gamma.3)+
        #   rnorm(n,0,sd=sigma_y)
        #
        #-------------------------------------------------------------#
        ######################### estimate delta_t   #################
        #-------------------------------------------------------------#
        ### delta(t) is estimated as the ratio of M(t) and W(t) based on the assumption
        ### that M(t) = delta(t)*W(t)
        delta_t.hat.smooth <- lowess(a, (colMeans(M_t)) / (colMeans(W_t)), f = 1 / 3)$y ## smoothed delta(t)
        delta_t.hat <- (colMeans(M_t)) / (colMeans(W_t)) ## non-smoothed delta(t)


        ## create new M_t by dividing the old one with estimated delta_t
        M_t.new <- M_t %*% diag(1 / c(delta_t.hat))
        M_t.news <- M_t %*% diag(1 / c(delta_t.hat.smooth))
        M_t.known <- M_t %*% diag(1 / c(delta_t))

        #-------------------------------------------------------------#
        ############################# Select k  #######################
        #-------------------------------------------------------------#

        BIC_value_naive <- sapply(5:k_max, L_BIC, Y = Y_norm, W = W_t, tau = p_sim, EF = Z, time_interval = a)
        k <- (5:k_max)[which.min(BIC_value_naive)]
        selected_k_naive[iter] <- k

        BIC_value_bench <- sapply(5:k_max, L_BIC, Y = Y_norm, W = X_t, tau = p_sim, EF = Z, time_interval = a)
        selected_k_bench[iter] <- (5:k_max)[which.min(BIC_value_bench)]

        k <- ceiling(n^(1 / 5)) + 2
        selected_k_n[iter] <- k
        bs2 <- bs(a, df = k, degree = 3, intercept = TRUE)

        #-------------------------------------------------------------#
        #################### basis expansion of M_t,W_t,X_t  ##########
        #-------------------------------------------------------------#

        W_i <- t(t(W_t) - colMeans(W_t)) %*% bs2 / length(a) # (n,k) matrix inner product
        X_i <- t(t(X_t) - colMeans(X_t)) %*% bs2 / length(a) # (n,k) matrix
        M_i <- t(t(M_t) - colMeans(M_t)) %*% bs2 / length(a) # (n,k) matrix
        M_i.new <- t(t(M_t.new) - colMeans(M_t.new)) %*% bs2 / length(a) # (n,k) matrix
        M_i.news <- t(t(M_t.news) - colMeans(M_t.news)) %*% bs2 / length(a) # (n,k) matrix
        M_i.known <- t(t(M_t.known) - colMeans(M_t.known)) %*% bs2 / length(a) # (n,k) matrix

        #-------------------------------------------------------------#
        #################### Benchmark regression model  ##############
        #-------------------------------------------------------------#

        # model = rq(Y_norm ~ X_i +EF + EF.b, tau=p_sim)
        model <- rq(Y_norm ~ X_i + Z, tau = p_sim)
        quantreg_X <- model$coefficients[2:(k + 1)]
        beta_X[, iter] <- crossprod(t(bs2), quantreg_X)
        beta_X_EF[, iter] <- model$coefficients[(k + 2)] ## estimate of EF when functional variable is true measurement
        beta_X_EF.b[, iter] <- model$coefficients[(k + 3)]

        #-------------------------------------------------------------#
        #################### Naive regression model  ##############
        #-------------------------------------------------------------#
        # model.W = rq(Y_norm ~ W_i +EF+ EF.b, tau=p_sim)
        model.W <- rq(Y_norm ~ W_i + Z, tau = p_sim)
        quantreg_W <- model.W$coefficients[2:(k + 1)]
        beta_naive[, iter] <- crossprod(t(bs2), quantreg_W)
        beta_W_EF[, iter] <- model.W$coefficients[(k + 2)] ## estimate of EF when ME in functional variable is ignored
        beta_W_EF.b[, iter] <- model.W$coefficients[(k + 3)]

        #-------------------------------------------------------------#
        ############ Get a reasonable estimate for Sigma_uu  #########
        #-------------------------------------------------------------#

        ##### estimated covariance matrix of X(t) ####
        ####### by assumption cov(W(t),M(t))/delta= Sigma_xx
        Sigma_xx <- try(cov(W_i, M_i), T)
        Sigma_xx.delta <- try(cov(W_i, M_i.new), T)
        Sigma_xx.deltas <- try(cov(W_i, M_i.news), T)
        Sigma_xx.delta.known <- try(cov(W_i, M_i.known), T)

        ### estimated covariance matrix of W(t) ####
        Sigma_ww <- var(W_i) ######## covariance matrix of observed surrogate var(W_ic) (centered)

        ###### estimate covariance matrix of measurement error U(t) ###
        Sigma_uu.delta <- try(Sigma_ww - Sigma_xx.delta, T) ###### covariance matrix of W|x from W=X+U
        Sigma_uu.delta <- try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.delta))), T)

        Sigma_uu <- try(Sigma_ww - Sigma_xx, T)
        Sigma_uu <- try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu))), T)

        Sigma_uu.deltas <- try(Sigma_ww - Sigma_xx.deltas, T) ####### covariance matrix of W|x from W=X+U
        Sigma_uu.deltas <- try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.deltas))), T)

        Sigma_uu.delta.known <- try(Sigma_ww - Sigma_xx.delta.known, T) ####### covariance matrix of W|x from W=X+U
        Sigma_uu.delta.known <- try(make.positive.definite(as.matrix(forceSymmetric(Sigma_uu.delta.known))), T)

        #---------------------------------------------------------------#
        ######################### SIME simulation step  #################
        #---------------------------------------------------------------#
        EF <- Z[, 1]
        EF.b <- Z[, 2]
        ########## Simulation step of the SIMEX procedure, see page 101 of Ray's book##############################
        B <- 100 #### number of repliates
        lambda <- seq(0.0001, 2.0001, .05) ### get a set of monotonically increasing small numbers

        #### when delta(t) is assumed to be 1 #####
        gamma.simex <- lapply(seq(1:B), function(b) {
            sapply(lambda, function(s) {
                set.seed(b + iter)
                U_b <- try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu, empirical = TRUE), T)
                W_lambda <- try(W_i + (sqrt(s) * U_b), T)

                model <- try(rq(Y_norm ~ W_lambda + EF + EF.b, tau = p_sim), T)
                try(model$coefficients, T)
            })
        })

        #### when delta(t) is estimated as raw ratio #####
        gamma_simex.delta <- lapply(seq(1:B), function(b) {
            sapply(lambda, function(s) {
                set.seed(b + iter)
                U_b <- try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.delta, empirical = TRUE), T)
                W_lambda <- try(W_i + (sqrt(s) * U_b), T)

                model <- try(rq(Y_norm ~ W_lambda + EF + EF.b, tau = p_sim), T)
                try(model$coefficients, T)
            })
        })

        #### when delta(t) is estimated as smoothed ratio #####
        gamma_simex.deltas <- lapply(seq(1:B), function(b) {
            sapply(lambda, function(s) {
                set.seed(b + iter)
                U_b.deltas <- try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.deltas, empirical = TRUE), T)
                W_lambda.deltas <- try(W_i + (sqrt(s) * U_b.deltas), T)

                model <- try(rq(Y_norm ~ W_lambda.deltas + EF + EF.b, tau = p_sim), T)
                try(model$coefficients, T)
            })
        })

        #### when delta(t) is known #####
        gamma_simex.delta.known <- lapply(seq(1:B), function(b) {
            sapply(lambda, function(s) {
                set.seed(b + iter)
                U_b.delta.known <- try(mvrnorm(n, rep(0, ncol(W_i)), Sigma_uu.delta.known, empirical = TRUE), T)
                W_lambda.delta.known <- try(W_i + (sqrt(s) * U_b.delta.known), T)

                model <- try(rq(Y_norm ~ W_lambda.delta.known + EF + EF.b, tau = p_sim), T)
                try(model$coefficients, T)
            })
        })

        #-------------------------------------------------------------#
        ######################### extrapolation step  #################
        #-------------------------------------------------------------#
        #### get average accross B ###
        gamma_simex.ave <- try(Reduce("+", gamma.simex) / B, T)
        gamma_simex.delta.ave <- try(Reduce("+", gamma_simex.delta) / B, T)
        gamma_simex.deltas.ave <- try(Reduce("+", gamma_simex.deltas) / B, T)
        gamma_simex.delta.known.ave <- try(Reduce("+", gamma_simex.delta.known) / B, T)


        gamma.t.simex <- (try(extra_quad((k + 3), gamma_simex.ave, lambda, -1), T)) ## all regression coefficients
        gamma.t.simex.delta <- (try(extra_quad((k + 3), gamma_simex.delta.ave, lambda, -1), T))
        gamma.t.simex.deltas <- (try(extra_quad((k + 3), gamma_simex.deltas.ave, lambda, -1), T))
        gamma.t.simex.delta.known <- (try(extra_quad((k + 3), gamma_simex.delta.known.ave, lambda, -1), T))



        #-------------------------------------------------------------#
        ######################### assume delta is 1  #################
        #-------------------------------------------------------------#
        beta_simex[, iter] <- try(crossprod(t(bs2), (gamma.t.simex[2:(k + 1)])), T)
        beta_naive_EF[, iter] <- try((gamma.t.simex[k + 2]), T)
        beta_naive_EF.b[, iter] <- try((gamma.t.simex[k + 3]), T)


        #-------------------------------------------------------------#
        ######################### delta estimated  ###############
        #-------------------------------------------------------------#

        beta_simex.delta[, iter] <- try(crossprod(t(bs2), (gamma.t.simex.delta[2:(k + 1)])), T)
        beta_naive_EF.delta[, iter] <- try((gamma.t.simex.delta[k + 2]), T)
        beta_naive_EF.delta.b[, iter] <- try((gamma.t.simex.delta[k + 3]), T)


        #-------------------------------------------------------------#
        ######################### delta estimated smoothed ############
        #-------------------------------------------------------------#


        beta_simex.deltas[, iter] <- try(crossprod(t(bs2), (gamma.t.simex.deltas[2:(k + 1)])), T)
        beta_naive_EF.deltas[, iter] <- try((gamma.t.simex.deltas[k + 2]), T)
        beta_naive_EF.deltas.b[, iter] <- try((gamma.t.simex.deltas[k + 3]), T)


        #-------------------------------------------------------------#
        ######################### delta known #######################
        #-------------------------------------------------------------#
        beta_simex.delta.known[, iter] <- try(crossprod(t(bs2), (gamma.t.simex.delta.known[2:(k + 1)])), T)
        beta_naive_EF.delta.known[, iter] <- try((gamma.t.simex.delta.known[k + 2]), T)
        beta_naive_EF.delta.known.b[, iter] <- try((gamma.t.simex.delta.known[k + 3]), T)


        print(iter)
    } ### end of simulation iterations


    #-------------------------------------------------------------#
    ####################### output of the function ################
    #-------------------------------------------------------------#

    re <- list(
        beta_X = beta_X, beta_simex = beta_simex, beta_naive = beta_naive,
        selected_k_naive = selected_k_naive, selected_k_bench = selected_k_bench,
        beta_simex.delta = beta_simex.delta, ## SIMEX estimator when delta is assumed to be 1
        beta_simex.delta.known = beta_simex.delta.known, ### SIMEX estimator of functional variable when delta is known
        beta_simex.deltas = beta_simex.deltas, ### SIMEX estimator of functional variable when delta is estimated
        beta_X_EF = beta_X_EF, beta_W_EF = beta_W_EF,
        beta_X_EF.b = beta_X_EF.b, beta_W_EF.b = beta_W_EF.b,
        beta_naive_EF = beta_naive_EF, #### Estimator of EF covariate when ME in functional variable is corrected by assume delta is 1
        beta_naive_EF.delta = beta_naive_EF.delta,
        beta_naive_EF.delta.known = beta_naive_EF.delta.known,
        beta_naive_EF.deltas = beta_naive_EF.deltas,
        beta_naive_EF.b = beta_naive_EF.b, #### Estimator of the binary EF covariate when ME in functional variable is corrected by assume delta is 1
        beta_naive_EF.delta.b = beta_naive_EF.delta.b,
        beta_naive_EF.delta.known.b = beta_naive_EF.delta.known.b,
        beta_naive_EF.deltas.b = beta_naive_EF.deltas.b,
        delta_t = delta_t.hat,
        delta_t.smooth = delta_t.hat.smooth
    ) #### Estimator of EF covariate when ME in functional variable is corrected by estimating delta

    return(re)
} ### end of simulation function

#-------------------------------------------------------------#
########################### Example ###########################
#-------------------------------------------------------------#


### run a 500-replications simulation
# res = main.delta(n=200,c= 0,sd_x=1.5,sd_w=0.75,p_sim=0.5,fi = "f1",fx = "f7",seeds=123,sim_liter=500)







cat("\n\n All the functions have been imported ------\n")
