

BQBayes.ME_SoFRFuncIndiv <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util


  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n <- length(Y)
  Zmod <- cbind(1, Z)
  #------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  #-------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")
  # p.mat


  # initial.values


  #--- Another version
  # initial.values.qr2


  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  # Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)

  Thet0 <- InitialValuesQR2(Y, W, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- runif(n = pn, -4, 4) # rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  # updateGam

  #--- Update Beta (error free covariates)
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  # Pgam <- P.mat(pn)*tauval
  Sig0.bet0 <- 100 * diag(pn0)
  InvSig0.bet0 <- 1.0 * as.inverse(Sig0.bet0)
  Mu0.bet0 <- numeric(pn0)
  # updates

  #--- Update Gamma individually
  #--- Prior parms
  # pn0 = ncol(Z) +1
  # Pgam <- P.mat(pn)*tauval
  InvSig0.gam0 <- P.mat(pn)
  Mu0.gam0 <- numeric(pn)
  # updates

  #---- update beta using the GAL density instead
  K0 <- P.mat(pn)
  SigmaProp <- solve(-Thet0$HessBet)
  diag(SigmaProp) <- abs(diag(SigmaProp))
  c0 <- .75
  SigmaProp <- as.positive.definite(as.symmetric.matrix(SigmaProp))
  # SigmaProp <- diag(pn); c0 = .5
  updateGamma <- function(Thet, Y, Zmod, K0, SigmaProp, c0, g0) {
    # g0 = 2
    idk <- Thet$Cik.e
    # Zmod <-  Z #cbind(1,Z)
    Ydi <- Y - Zmod %*% Thet$BetaZ #- Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si - Thet$Ak[Thet$Cik.e]*Thet$nui  #Thet$muk.e[Thet$Cik.e]
    ei_old <- Ydi - Thet$X %*% Thet$Gamma

    Gam_newk <- TruncatedNormal::rtmvnorm(n = 1, mu = Thet$Gamma, sigma = c0 * SigmaProp, lb = rep(-g0, length(Thet$Gam)), ub = rep(g0, length(Thet$Gam)))
    ei_new <- Ydi - Thet$X %*% Gam_newk

    logLkold <- sum(log(mapply(fGal.p0, e = ei_old, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p = Thet$p[idk], tau0 = Thet$tau0) + .Machine$double.eps))
    -.5 * quad.form(K0 * (1 / Thet$siggam), Gam_newk) - TruncatedNormal::dtmvnorm(x = Gam_newk, mu = Thet$Gamma, sigma = c0 * SigmaProp, lb = rep(-g0, length(Thet$Gam)), ub = rep(g0, length(Thet$Gam)), log = T)
    # LaplacesDemon::dmvnp(x = Gam_newk, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)

    #  (sqrt(Thet$sig2k.e[k]), shape =a.sig,scale = b.sig, log=T) + dunif(Thet$gamk[k], min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
    # - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

    logLknew <- sum(log(mapply(fGal.p0, e = ei_new, sig2 = sqrt(Thet$sig2k.e[idk]), gam = Thet$gamk[idk], p = Thet$p[idk], tau0 = Thet$tau0) + .Machine$double.eps))
    -.5 * quad.form(K0 * (1 / Thet$siggam), Thet$Gamma) - TruncatedNormal::dtmvnorm(x = Thet$Gamma, mu = Gam_newk, sigma = c0 * SigmaProp, lb = rep(-g0, length(Thet$Gam)), ub = rep(g0, length(Thet$Gam)), log = T)
    #+ LaplacesDemon::dmvnp(x = Thet$Gamma, mu = rep(0, length(Thet$Gamma)), Omega = K0*(1/Thet$siggam), log = T)
    #+ dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = 0.95*Thet$gamL , max = 0.95*Thet$gamU, log = T)
    #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

    uv <- logLknew - logLkold
    jump <- runif(n = 1)
    if (log(jump) < uv) {
      res <- c(1, Gam_newk)
    } else {
      res <- c(0, Thet$Gamma)
    }
    res
  }
  updateGamma <- cmpfun(updateGamma)



  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  # varxi <- sqrt(c(5,5,5))
  # updates
  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = pn) # length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  MCMCSample$AcceptGam <- rep(NA, Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")

  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)

    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet = Thet))
    # Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))

    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)

    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A = A, B = B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for (k in 1:Kw) {
      nk[k] <- sum(Thet$Cik.w == k)
    }
    # update the Pis
    alpha.wnew <- alpha.W / Kw + nk
    Thet$Pik.w <- rdirch(n = 1, alpha.wnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w)
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for (k in 1:nrow(Thet$Muk.m)) {
      nk[k] <- sum(Thet$Cik.m == k)
    }
    # update the Pis
    alpha.mnew <- alpha.m / Kw + nk
    Thet$Pik.m <- rdirch(n = 1, alpha.mnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # if(iter%%50 == 0){
    Thet$Betadel <- rep(1, ncol(Mn)) # updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter, ] <- Thet$Betadel
    # }
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    # Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    # Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    # Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #

    Thet$BetaZ <- updateGamBeta(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0) # Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Tep <- updateGamma(Thet, Y, Zmod, K0, SigmaProp, c0, g0)
    # print(Tep[length(Tep)])
    GamAccep <- Tep[1]
    Thet$Gamma <- Tep[-1] # updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0)
    # updateGamma(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0)  # Tp0[-c(1:ncol(Zmod))]

    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    MCMCSample$AcceptGam[iter] <- GamAccep

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}




#-- using FUI approach
#-- Fixed x (we know X) Fx
# sig02.del = 0.1; tauval = .1;  Delt= rep(1, ncol(M)); alpx=1; tau0 = 0.5
BQBayes.ME_SoFRFuncIndivFxFUI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")




  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = pn #min(15, ceiling(length(Y)^{1/3.8})+4)
  n <- length(Y)
  Zmod <- cbind(1, Z)
  #------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  #-------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W[, , 1] %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")
  # p.mat


  # initial.values


  #--- Another version
  # initial.values.qr2


  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  # Thet0 <- InitialValuesQR(Y, W, M, Z, df0=3, pn, Gx=1, a, tau0)
  W01 <- apply(W, c(1, 2), mean)

  Thet0 <- InitialValuesQR2(Y, W01, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- runif(n = pn, -4, 4) # rnorm(n=pn) #as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  # updateGam

  #--- Update Beta (error free covariates)
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  # Pgam <- P.mat(pn)*tauval
  Sig0.bet0 <- 100 * diag(pn0)
  InvSig0.bet0 <- 1.0 * as.inverse(Sig0.bet0)
  Mu0.bet0 <- numeric(pn0)
  # updates

  #--- Update Gamma individually
  #--- Prior parms
  # pn0 = ncol(Z) +1
  # Pgam <- P.mat(pn)*tauval
  InvSig0.gam0 <- P.mat(pn)
  Mu0.gam0 <- numeric(pn)
  # updates

  #---- update beta using the GAL density instead
  K0 <- P.mat(pn)
  SigmaProp <- solve(-Thet0$HessBet)
  diag(SigmaProp) <- abs(diag(SigmaProp))
  c0 <- .5
  SigmaProp <- as.positive.definite(as.symmetric.matrix(SigmaProp))
  # SigmaProp <- diag(pn); c0 = .5
  # updates



  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  # varxi <- sqrt(c(5,5,5))
  # updates
  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = pn) # length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)
  MCMCSample$AcceptGam <- rep(NA, Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")
  # Thet$X <-
  Thet$X <- (as.matrix(FUI(model = "gaussian", smooth = TRUE, W, silent = FALSE)) %*% bs2) / length(a)
  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    #
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))
    #
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    #
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    #
    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    # Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn#UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    # #--- Update Pik.w
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    # Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    # Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    # Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #

    Thet$BetaZ <- updateGamBeta(Thet, Y, Zmod, InvSig0.bet0, Mu0.bet0) # Tp0[c(1:ncol(Zmod))] #c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Tep <- updateGamma(Thet, Y, Zmod, K0, SigmaProp, c0, g0)
    # print(Tep[length(Tep)])
    GamAccep <- Tep[1]
    Thet$Gamma <- Tep[-1] # updateGamma(Thet, Y, Zmod, K0,SigmaProp, c0)
    # updateGamma(Thet, Y, Zmod, InvSig0.gam0, Mu0.gam0)  # Tp0[-c(1:ncol(Zmod))]

    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e
    MCMCSample$AcceptGam[iter] <- GamAccep

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$XFui <- Thet$X
  MCMCSample
}



#---- Simulated tau
#-- Using the rtmvnorm approach for the truncated multivariate normal
#--- Update Beta(t) and beta_z together ()
#-- Fix X
BQBayes.ME_SoFRFuncSimFx <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util


  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")


  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  # varxi <- c(10,10,10)*.25
  # varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10, 10, 10) * .35 #-- high acceptance rate
  varxi <- c(10, 10, 10) * tunprop ##  tunprop
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 ## This works well
    ab <- 10
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat) {
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 <- matrix(0, nrow = K, ncol = ncol(Mat))
    for (k in 1:K) {
      Res0[k, ] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k, ], ncol = J, byrow = F))))
    }
    Res0
  }


  #----------------------------------------------
  # Update things related to Wi
  #--------------------------------------------
  # Piw
  # Cik.w
  #--- Update Pik.x
  # alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W) {
    Ku <- nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for (k in 1:Ku) {
      nk[k] <- sum(Thet$Cik.w == k)
    }

    # update the Pis
    alpha.wnew <- alpha.W / length(Thet$muk.w) + nk

    rdirichlet(n = 1, alpha.wnew)
  }

  #--- Update Ci.k mixture of normals
  dmvnormVec <- function(x, Mean, SigmaVc) {
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for (k in 1:Ku) {
      Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k, ]), sigma = as.symmetric.matrix(matrix(SigmaVc[k, ], ncol = length(x), byrow = F)))
    }
    Val
  }
  # dmvnormVec <- Vectorize(dmvnormVec,"k")
  # dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)

  updateCik.w <- function(i, Thet, W) { # mixture of normals
    # Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    # kw = nrow(Thet$Muk.w)
    xi <- c(W[i, ] - Thet$X[i, ])
    # pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k <- Thet$Pik.w * dmvnormVec(x = xi, Mean = Thet$Muk.w, SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n = 1, size = 1, prob = pi.k) == 1)
  }

  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- Xn
  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)

    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet = Thet))
    # Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))

    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)

    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    # Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for (k in 1:Kw) {
      nk[k] <- sum(Thet$Cik.w == k)
    }
    # update the Pis
    alpha.wnew <- alpha.W / Kw + nk
    Thet$Pik.w <- rdirch(n = 1, alpha.wnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w)
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for (k in 1:nrow(Thet$Muk.m)) {
      nk[k] <- sum(Thet$Cik.m == k)
    }
    # update the Pis
    alpha.mnew <- alpha.m / Kw + nk
    Thet$Pik.m <- rdirch(n = 1, alpha.mnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # if(iter%%50 == 0){
    Thet$Betadel <- rep(1, ncol(Mn)) # updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter, ] <- Thet$Betadel
    # }
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#--  using W (average over days) and M
BQBayes.ME_SoFRFuncSim <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util


  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")

  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    expr1 <- quote(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
    res <- R.utils::withTimeout(expr1, substitute = F, timeout = 30, onTimeout = "silent")
    if (is.null(res)) {
      # res = c(c(mvtnorm::rmvnorm(n=1, mean = Mu.gam[1:pn0], sigma = as.positive.definite(Sig.gam)[1:pn0,1:pn0])),
      #        c( TruncatedNormal::rtmvnorm(n=1, mu = Mu0.gam0[-c(1:pn0)], sigma = as.positive.definite(Sig.gam)[-c(1:pn0),-c(1:pn0)], lb =c(rep(-g0,length(Mu.gam)-ncol(Zmod))),
      #                                     ub = c(rep(g0,length(Mu.gam)-ncol(Zmod))) ) ) )
      res <- c(tmvtnorm::rtmvnorm(
        n = 1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam), lower = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
        upper = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod))), algorithm = "gibbs", burn.in.samples = 100
      ))
    }

    # expr2 <- quotec(tmvtnorm::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb =c(rep(-Inf, ncol(Zmod)),rep(-g0,length(Mu.gam)-ncol(Zmod))),
    #                             ub = c(rep(Inf, ncol(Zmod)),rep(g0,length(Mu.gam)-ncol(Zmod)))   ))

    res
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # val <-
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  varxi <- c(10, 10, 10) * .45 ## high aceeptance tunprop
  varxi <- c(10, 10, 10) * tunprop ##  tunprop
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 # THis works well
    ab <- 10 # THis works well
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices


  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")

  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)

    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet = Thet))
    # Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))

    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)

    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    if (iter %% 100 == 0) {
      Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A = A, B = B)
    }
    # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for (k in 1:Kw) {
      nk[k] <- sum(Thet$Cik.w == k)
    }
    # update the Pis
    alpha.wnew <- alpha.W / Kw + nk
    Thet$Pik.w <- rdirch(n = 1, alpha.wnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w)
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for (k in 1:nrow(Thet$Muk.m)) {
      nk[k] <- sum(Thet$Cik.m == k)
    }
    # update the Pis
    alpha.mnew <- alpha.m / Kw + nk
    Thet$Pik.m <- rdirch(n = 1, alpha.mnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # if(iter%%50 == 0){
    Thet$Betadel <- rep(1, ncol(Mn)) # updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter, ] <- Thet$Betadel
    # }
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxFUI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  #------------------------------------
  #------------
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")



  F2Cmp <- cmpfun(F2)

  GamF <- function(gam, p0) {
    (2 * pnorm(-abs(gam)) * exp(.5 * gam^2) - p0)^2
  }

  # optimize(GamF,interval = c(-30,30), p0=1 - 0.05)
  GamBnd <- function(p0) {
    Re1 <- optimize(GamF, interval = c(-30, 30), p0 = 1 - p0)
    Re2 <- optimize(GamF, interval = c(-30, 30), p0 = p0)

    c(-abs(Re1$minimum), abs(Re2$minimum), Re1$objective, Re2$objective)
  }

  #--------------------------------------------
  # gal

  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W[, , 1] %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")


  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W, c(1, 2), mean), M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5
  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  # varxi <- c(10,10,10)*.25
  # varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10, 10, 10) * .35 ## -- High acceptance rate
  varxi <- c(10, 10, 10) * tunprop ## --
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 ## This works well
    ab <- 10
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices

  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- (as.matrix(FUI(model = "gaussian", smooth = T, W, silent = FALSE)) %*% bs2) / length(a)
  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    #
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))
    #
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    #
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    # Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X

  MCMCSample
}



#--- Naive approach using W
#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxNaiveW <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  #------------------------------------
  #------------
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W[, , 1] %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")

  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W, c(1, 2), mean), M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  # varxi <- c(10,10,10)*.25
  # varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10, 10, 10) * .35 ## -- High acceptance rate
  varxi <- c(10, 10, 10) * tunprop ## --
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 ## This works well
    ab <- 10
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices

  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")
  Thet$X <- (apply(W, c(1, 2), mean) %*% bs2) / length(a)
  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    #
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))
    #
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    #
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    # Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X

  MCMCSample
}



#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxFSMI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  #------------------------------------
  #------------
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util
  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W[, , 1] %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")

  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, apply(W, c(1, 2), mean), M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  # varxi <- c(10,10,10)*.25
  # varxi <- c(10,10,10)*.15 ## high acceptance rate
  varxi <- c(10, 10, 10) * .35 ## -- High acceptance rate
  varxi <- c(10, 10, 10) * tunprop ## --
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 ## This works well
    ab <- 10
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices

  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = nrow(Wn))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = nrow(Wn))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")
  # Thet$X <- (as.matrix(FUI(model="gaussian",smooth=T, W,silent = FALSE)) %*%bs2)/length(a)
  Thet$X <- (FSMI(model = "gaussian", cov.model = "us", W) %*% bs2) / length(a)
  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    #
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))
    #
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    #
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    # if(iter%%10 == 0){
    # Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample$Xpred <- Thet$X

  MCMCSample
}

#----- Case of replicated days
# sig02.del = 0.1; tauval = .1;  Delt= rep(1, ncol(M)); alpx=1; tau0 = 0.5
BQBayes.ME_SoFRFuncSimWrep <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  func <- function(Pik, Pi0, n = 1, size = 1) {
    which.max(rmultinom(n = 1, size = 1, prob = exp(Pik + abs(max(Pik))) * Pi0))
    # which.max(rmultinom(n=1,size=1,prob = Pik))
  }
  funcCmp <- cmpfun(func)

  F2 <- function(Pik, Pi0, n = 1, size = 1) {
    apply(Pik, 1, funcCmp, Pi0 = Pi0)
  }

  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- array(NA, c(dim(W)[1], pn, n)) # (W%*%bs2)/length(a)

  #--- Reduce the dimension of the data
  for (i in 1:n) {
    Wn[, , i] <- (W[, , i] %*% bs2) / length(a)
  }


  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")


  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, t(apply(W, c(2, 3), mean)), apply(M, c(1, 2), mean), Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- t(apply(Wn, c(2, 3), mean)) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # val <-
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  varxi <- c(10, 10, 10) * .45 ## high aceeptance tunprop
  varxi <- c(10, 10, 10) * tunprop ##  tunprop
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 # THis works well
    ab <- 10 # THis works well
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices

  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w) {
    Ku <- length(Thet$Pik.w)
    J <- ncol(Thet$X)
    nrep <- dim(W)[1]
    Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
    SigK <- list() #-- store the covariances
    SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
    Yi.tld <- apply(W, 2, c) - matrix(rep(c((Thet$X)), each = dim(W)[1]), ncol = ncol(Thet$X), byrow = F) # Thet$X
    Yi.tld <- W - array(matrix(rep(c((Thet$X)), each = dim(W)[1]), ncol = ncol(Thet$X), byrow = F), c(dim(W)[1], ncol(Thet$X), nrow(Thet$X))) ## This is now a array  where the rows of X are replicated nrep times in the array

    # Muk <- matrix(0, ncol=J,nrow=Ku)
    SigK <- list()
    for (k in 1:Ku) {
      idk <- which(Thet$Cik.w == k)
      nk <- length(idk)
      if (length(idk) > 0) {
        Yi0 <- apply(Yi.tld[, , idk], 2, c) # matrix(c(Yi.tld[idk,]), nrow= nk, byrow=F)
        TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k, ], ncol = J, byrow = F)))
        SigK[[k]] <- as.inverse(as.symmetric.matrix(nrep * nk * TpSi + inSig0.w)) ## sum of matrices inverse
        SigKWg <- SigKWg + (Thet$Pik.w[k] * Thet$Pik.w[k]) * SigK[[k]] # J x J
        # Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(colSums(Yi.tld[idk,])) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
        Mu.k[k, ] <- c(SigK[[k]] %*% (crossprod(TpSi, c(colSums(Yi0))) + inSig0.w %*% Mu0.w)) ## some of mat
        # Mu.k[k,] <- SigK[[k]]%*%(crossprod(TpSi, c(apply(t(Yi.tld[idk,]), 2, sum)) ) + solve(Sig0.w)%*%Mu0.w)      ## some of mat
      } else {
        SigK[[k]] <- Sig0.w
        SigKWg <- SigKWg + (Thet$Pik.w[k] * Thet$Pik.w[k]) * SigK[[k]] # J x J
        Mu.k[k, ] <- Mu0.w
      }
    }
    #--- we impose some constrainsts on the mean
    SIg0 <- bdiag(SigK) # create a block diagonal matrix
    SIGR0 <- NULL
    for (k in 1:Ku) {
      SIGR0 <- rbind(SIGR0, Thet$Pik.w[k] * SigK[[k]])
    } ## K*J x J
    MUR0 <- colSums(diag(Thet$Pik.w) %*% Mu.k) # return a evector of length J

    MURFin <- c(t(Mu.k)) - SIGR0 %*% solve(SigKWg) %*% MUR0 ## J*K
    SIGR0fin <- 1.0 * as.matrix(nearPD(SIg0 - SIGR0 %*% solve(SigKWg) %*% t(SIGR0), doSym = T)$mat) ## J*K x J*K

    #--- Simulate K-1 vectors from the degenerate dist.
    id <- 1:(J * (Ku - 1))
    SimMU.k <- matrix(c(mvtnorm::rmvnorm(n = 1, mean = MURFin[id], sigma = SIGR0fin[id, id])), ncol = J, byrow = T) ## (K-1) x J
    SimMU.k1 <- -colSums(diag(Thet$Pik.w[-Ku]) %*% SimMU.k) / Thet$Pik.w[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
    as.matrix(rbind(SimMU.k, SimMU.k1)) ## k * J
  }
  updateMuk.w <- cmpfun(updateMuk.w)

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(apply(Wn, 2, c)) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  updateSigk.w <- function(Thet, W, nu0.w, Psi0.w) {
    Ku <- nrow(Thet$Muk.w)
    nrep <- dim(W)[1]
    m <- dim(W)[2]
    al <- numeric(Ku)
    bet <- numeric(Ku)
    # Wtl <- W -  Thet$X - Thet$Muk.w[Thet$Cik.w,]
    Wtl <- W - array(matrix(rep(c((Thet$X + Thet$Muk.w[Thet$Cik.w, ])), each = dim(W)[1]), ncol = ncol(Thet$X), byrow = F), c(dim(W)[1], ncol(Thet$X), nrow(Thet$X))) ## An array (nrep, Pn, n)
    Res <- matrix(0, ncol = m * m, nrow = Ku)
    for (i in 1:Ku) {
      id <- which(Thet$Cik.w == i)
      nk <- length(id)
      if (nk > 0) {
        nun <- nu0.w + nk * nrep
        # Yi0 = matrix(c(Wtl[id,]), nrow= nk, byrow=F)
        Yi0 <- apply(Wtl[, , id], 2, c)
        Psin.w <- as.positive.definite(Psi0.w + crossprod(Yi0))
        Res[i, ] <- c(rinvwishart(nun, Psin.w))
      } else {
        Res[i, ] <- c(rinvwishart(nu0.w, Psi0.w))
      }
    }
    Res
  }
  updateSigk.w <- cmpfun(updateSigk.w)

  #----------------------------------------------
  # Update things related to M
  #--------------------------------------------

  #--- update Pik.m for epsilon_i
  # alpha.m = 1.0

  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(t(apply(W, c(2, 3), mean)))) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (t(apply(Wn, c(2, 3), mean)))) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))


  ## This approach of simulating Xi only uses W, M, and X

  udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A = A, B = B) {
    J <- ncol(Thet$X)
    n <- length(Y)
    # Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
    Ytl <- c((Y - Zmod %*% Thet$BetaZ - Thet$Ak[Thet$Cik.e] * Thet$nui - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si) / (Thet$Bk[Thet$Cik.e] * sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$nui)) # Thet$Gam
    # Wtl = Wn - Thet$Muk.w[Thet$Cik.w,]

    Wtl <- Wn - array(matrix(rep(c((Thet$Muk.w[Thet$Cik.w, ])), each = dim(Wn)[1]), ncol = ncol(Thet$X), byrow = F), c(dim(Wn)[1], ncol(Thet$X), nrow(Thet$X))) ## This is an array (nrep, pn,n)
    # Mtl = Mn - Thet$Muk.m[Thet$Cik.m,] #%*%diag(Thet$Betadel)
    Res <- matrix(0, ncol = J, nrow = n)
    Sigw <- matrix(0, J, J)
    Sigm <- matrix(0, J, J)
    Sigx <- matrix(0, J, J)
    GamMat <- tcrossprod(c(Thet$Gamma))

    for (i in 1:n) {
      ie <- Thet$Cik.e[i]
      iw <- Thet$Cik.w[i]
      im <- Thet$Cik.m[i]
      ix <- Thet$Cik.x[i]

      # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
      # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
      # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#

      Sigw <- matrix(Thet$InvSigk.w[iw, ], ncol = J, byrow = F)
      Sigm <- matrix(Thet$InvSigk.m[im, ], ncol = J, byrow = F) * 0
      Sigx <- matrix(Thet$InvSigk.x[ix, ], ncol = J, byrow = F)

      # Thet$InvSigk.m

      # Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
      # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
      Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat / (Thet$Bk[ie] * sqrt(Thet$sig2k.e[ie]) * Thet$nui[i]) + (dim(Wn)[1]) * Sigw + Sigm * 0 + Sigx))) # quad.form(Sigm, diag(Thet$Betadel))
      # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel))

      # Mu <- c(Sig%*%(c(Thet$Gamma)*Ytl[i] + crossprod(Sigw, matrix(colSums(Wtl[,,i])), ncol=1) + 0*crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,])));
      Mu <- c(Sig %*% (c(Thet$Gamma) * Ytl[i] + crossprod(Sigw, matrix(colSums(Wtl[, , i])), ncol = 1) + crossprod(Sigx, Thet$Muk.x[ix, ])))

      # Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,])));

      # c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
      # Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
      Res[i, ] <- c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu, sigma = Sig, lb = rep(A, J), ub = rep(B, J)))
    }
    Res
  }
  # udapateXi <- Vectorize(udapateXi,"i")
  udapateXi2 <- cmpfun(udapateXi2)




  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = length(Y))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  cat(dim(Wn), "\n")
  cat("\n #--- MCMC is starting ----> \n")

  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)

    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet = Thet))
    # Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))

    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)

    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    if (iter %% 100 == 0) {
      Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A = A, B = B)
    }
    # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for (k in 1:Kw) {
      nk[k] <- sum(Thet$Cik.w == k)
    }
    # update the Pis
    alpha.wnew <- alpha.W / Kw + nk
    Thet$Pik.w <- rdirch(n = 1, alpha.wnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # Thet$Cik.w <- F2Cmp(CikWRep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w)) #F2LCmp
    # W0 <- CikWrep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w)
    # W0 <- CikW(t(apply(Wn, c(2,3), mean)), Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w)
    Thet$Cik.w <- F2Cmp(CikWrep(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w), Thet$Pik.w) # F2LCmp
    # Thet$Cik.w <- F2Cmp(CikWRepVec(c(dim(Wn)), c(Wn), Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w), )
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w)
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # #Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # }
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}



#--- replicate W
#-- THis function will perform the estimation
#-- replicates values of W

BQBayes.ME_SoFRFuncSimRepW <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util


  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  n <- length(Y)
  Zmod <- cbind(1, Z)
  pn <- c(5, 6, 20, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)

  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")
  P.mat <- function(K) {
    # penalty matrix
    D <- diag(rep(1, K))
    D <- diff(diff(D))
    P <- t(D) %*% D #+ diag(rep(1e-6, K))
    return(P)
  }


  # initial.values

  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
    c(TruncatedNormal::rtmvnorm(
      n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = c(rep(-Inf, ncol(Zmod)), rep(-g0, length(Mu.gam) - ncol(Zmod))),
      ub = c(rep(Inf, ncol(Zmod)), rep(g0, length(Mu.gam) - ncol(Zmod)))
    ))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  updateSigGam0 <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    val <- nimble::rinvgamma(n = 1, shape = al, scale = bet)
    ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  al0 <- 0.001 # 1 #1
  bet0 <- 0.001 # 0.005 #.005
  updateSigGam <- function(Thet, al0, bet0, order = 2) {
    K <- length(Thet$Gamma)
    al <- al0 + .5 * (K - order)
    bet <- .5 * quad.form(P.mat(K), Thet$Gamma) + bet0
    # 1/rgamma(n = 1, shape = al, rate = bet)
    # val <-
    # rgamma(n = 1, shape = al, scale = bet)
    nimble::rinvgamma(n = 1, shape = al, scale = bet)
    # ifelse(val > 20, nimble::rinvgamma(n = 1, shape = al0, scale = bet0), val)
    # tb = 10
    # 1/heavy::rtgamma(n=1,shape=al,scale = bet, t = tb)
    # tb = 1000
    # 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb, min = 1)
    # tb = .01
    # ifelse( qgamma(.95, shape= al, rate = bet) > tb, 1/tb , 1/cascsim::rtgamma(n=1,shape=al, scale = 1/bet, max = tb))
  }


  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  varxi <- c(10, 10, 10) * .45 ## high aceeptance tunprop
  varxi <- c(10, 10, 10) * tunprop ##  tunprop
  # varxi <- sqrt(c(5,5,5))
  updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
    a0 <- 0.1
    ab <- 5 # THis works well
    ab <- 10 # THis works well
    # varxi = 5
    idk <- which(Thet$Cik.e == k)
    if (length(idk) > 0) {
      ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
      Val <- TruncatedNormal::rtmvnorm(n = 1, mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU))


      sig_newk <- Val[1] # truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
      gam_newk <- Val[2] # truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

      logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), mu = c(sig_newk, gam_newk), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
      + nimble::dinvgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
        - dtmvnorm(c(sig_newk, gam_newk), mu = c(sqrt(Thet$sig2k.e[k]), Thet$gamk[k]), sigma = Sdsig.prop * varxi[k], lb = c(a0, 0.95 * Thet$gamL), ub = c(ab, .95 * Thet$gamU), log = TRUE))
      #- log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k] , sd = sdgam.prop, a = 0.95*Thet$gamL, b = .95*Thet$gamU) + .Machine$double.eps))

      #  logLknew <- sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew) + .Machine$double.eps))
      #  + dgamma(sig_newk, shape =a.sig,scale = b.sig, log=T) + dunif(gam_newk, min = Thet$gamL , max = Thet$gamU, log = T)
      # print(sig_newk)
      uv <- logLknew - logLkold
      if (is.numeric(uv)) {
        jump <- runif(n = 1)
        indic <- 1 * (log(jump) < uv)
        Res <- indic * c(sig_newk^2, gam_newk, pnew, log(jump) < uv, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew)) + (1 - indic) * c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], log(jump) <= uv, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k])
      } else {
        Res <- c(Thet$sig2k.e[k], Thet$gamk[k], Thet$p[k], 0, Thet$Ak[k], Thet$Bk[k], Thet$Ck[k]) # log(jump) <= uv
      }

      if (is.na(uv) || is.nan(uv)) {
        sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
        gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
        pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
        Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
      }
    } else {
      sig_newk <- nimble::rinvgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }

    Res
  }
  updatesig_gamHes <- Vectorize(updatesig_gamHes, "k")


  #--- THis function will inverse the covariance matrices
  # Mat : is a mtrix of K X (p*p) a matrix of K  pxp matrices
  InvCov <- function(Mat) {
    K <- nrow(Mat)
    J <- round(sqrt(ncol(Mat)))
    Res0 <- matrix(0, nrow = K, ncol = ncol(Mat))
    for (k in 1:K) {
      Res0[k, ] <- c(as.inverse(as.symmetric.matrix(matrix(Mat[k, ], ncol = J, byrow = F))))
    }
    Res0
  }


  #----------------------------------------------
  # Update things related to Wi
  #--------------------------------------------
  # Piw
  # Cik.w
  #--- Update Pik.x
  # alpha.W = alpha.w = 1.0
  updatePik.w <- function(Thet, alpha.W) {
    Ku <- nrow(Thet$Muk.w)
    nk <- numeric(Ku)
    for (k in 1:Ku) {
      nk[k] <- sum(Thet$Cik.w == k)
    }

    # update the Pis
    alpha.wnew <- alpha.W / length(Thet$muk.w) + nk

    rdirichlet(n = 1, alpha.wnew)
  }

  #--- Update Ci.k mixture of normals
  dmvnormVec <- function(x, Mean, SigmaVc) {
    Ku <- nrow(Mean)
    Val <- numeric(Ku)
    for (k in 1:Ku) {
      Val[k] <- mvtnorm::dmvnorm(x = x, mean = c(Mean[k, ]), sigma = as.symmetric.matrix(matrix(SigmaVc[k, ], ncol = length(x), byrow = F)))
    }
    Val
  }
  # dmvnormVec <- Vectorize(dmvnormVec,"k")
  # dmvnormVec(x = rep(0,6), Mn=Thet$Muk.w,SigmaVc = Thet$Sigk.w)

  updateCik.w <- function(i, Thet, W) { # mixture of normals
    # Sig <- matrix(Thet$Sigk.w[id,], ncol = ncol(W), byrow = F)
    # kw = nrow(Thet$Muk.w)
    xi <- c(W[i, ] - Thet$X[i, ])
    # pi.k  = Thet$Pik.w*mapply(dmvnormVec, k = 1:Kw, MoreArgs = list(x = xi, Mean = Thet$Muk.w , SigmaVc = Thet$Sigk.w))
    pi.k <- Thet$Pik.w * dmvnormVec(x = xi, Mean = Thet$Muk.w, SigmaVc = Thet$Sigk.w) + .Machine$double.eps
    which(rmultinom(n = 1, size = 1, prob = pi.k) == 1)
  }

  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = length(Y))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")

  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    #----------------------------------------------
    # Update things related to X
    #--------------------------------------------
    #--- update Pik.m for epsilon_i
    Thet$Pik.x <- updatePik.x(Thet, alpha.x)

    #--- update Cik.m for epsilon_i based on mixture of normals
    Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet = Thet))
    # Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))

    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)

    Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    Thet$InvSigk.x <- InvCov(Thet$Sigk.x)

    # Ytd = Y - Zmod%*%Thet$BetaZ
    if (iter %% 100 == 0) {
      Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A = A, B = B)
    }
    # Xn #
    # Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)

    # Thet$X <- Xn #UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])

    # }
    # dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))

    # Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta

    # Thet$X <-
    #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    #---------------------------------------------
    #--- update things related to W
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Kw)
    for (k in 1:Kw) {
      nk[k] <- sum(Thet$Cik.w == k)
    }
    # update the Pis
    alpha.wnew <- alpha.W / Kw + nk
    Thet$Pik.w <- rdirch(n = 1, alpha.wnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w)
    Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    #---------------------------------------------
    #--- update things related to M
    #---------------------------------------------
    #--- Update Pik.w
    # updatePik.w <- function(Thet, alpha.W){
    nk <- numeric(Km)
    for (k in 1:nrow(Thet$Muk.m)) {
      nk[k] <- sum(Thet$Cik.m == k)
    }
    # update the Pis
    alpha.mnew <- alpha.m / Kw + nk
    Thet$Pik.m <- rdirch(n = 1, alpha.mnew)

    #--- Update Ci.k mixture of normals
    # Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    Thet$Cik.m <- F2Cmp(CikW(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # if(iter%%50 == 0){
    Thet$Betadel <- rep(1, ncol(Mn)) # updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    MCMCSample$Betdel[iter, ] <- Thet$Betadel
    # }
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0, g0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- updateSigGam(Thet, al0, bet0, order = 2) # sig02.del# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}


#--- FInal function to be run
meth <- c("Wrep", "FUI", "Wnaiv")


FinalMeth <- function(meth, Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop) {
  switch(meth,
    Wrep = BQBayes.ME_SoFRFuncSimWrep(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop),
    FUI = BQBayes.ME_SoFRFuncSimFxFUI(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop),
    Wnaiv = BQBayes.ME_SoFRFuncSimFxNaiveW(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0, X, g0, tunprop)
  )
}




#--- This will estimate the quantile assuming W or true X (No measurrement error)
BQBayes.SoFRFunc <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X) {
  # Basis
  met <- c("f1", "f2", "f3", "f4", "f5")


  # util


  cat("--- set some initial values ")
  #-------------------------------------
  #---- Parameters settings
  #-------------------------------------
  # pn = c(3, 6, 10, 15)[which(n == c(100, 200,500,1000))]
  # pn = c(5, 6, 20, 15)[which(n == c(100, 200,500,1000))]
  pn <- c(5, 6, 15, 20)[which(n == c(100, 200, 500, 1000))]
  alpha.e <- alpx
  alpha.w <- alpha.W <- alpx
  alpha.x <- alpha.X <- alpx
  alpha.m <- alpx
  pn <- pn # min(15, ceiling(length(Y)^{1/3.8})+4)
  n <- length(Y)
  Zmod <- cbind(1, Z)
  #-------------------------------------
  #--- Transform the data
  #--------------------------------------
  Delthat <- Delt # c((colMeans(M)/(colMeans(W)))) # lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7)$y #  lowess(a, abs(colMeans(M)/colMeans(W)), f=1/7) #
  Mn_Mod <- M %*% diag(1 / Delthat)
  bs20 <- bs(a, df = pn, intercept = T)
  bs20 <- bs(a, df = pn, intercept = T, degree = 2)
  bs2 <- B.basis(a, as.numeric(c(0, attr(bs20, "knots"), 1)))
  Mn <- (Mn_Mod %*% bs2) / length(a) # (M%*%bs2)/length(a)
  Wn <- (W %*% bs2) / length(a)
  Xn <- (X %*% bs2) / length(a)

  cat("\n #--- Initial values...\n")
  # p.mat


  # initial.values

  # Thet0 <- InitialValues(Y, W, M, Z, df0=3, pn, Gx=2, a, Respfr)
  Thet0 <- InitialValuesQR(Y, W, M, Z, df0 = 3, pn, Gx = 1, a, tau0)
  #----------------------------------------------
  #---  Provide the constants
  #----------------------------------------------
  # alpha.e = 1.0
  # alpha.w = alpha.W = 1.0
  # alpha.x = alpha.X = 1.0
  # alpha.m = 1.0
  # Kvalues
  # Kc <- 5

  Kx <- Thet0$Kx ## number of X  cluster
  Ke <- Thet0$Ke ## number of ei cluster
  Km <- Thet0$Km ## number of ei cluster
  Kw <- Thet0$Kw ## number of omega_i cluster

  #---------------------------------------------
  cat("\n#---- Initialized parameters ---> \n")
  #---------------------------------------------
  Thet <- list()
  Thet$Delta <- Thet0$Delta #- The intercepts vector of length J
  Thet$Gamma <- rnorm(n = pn) # as.numeric(Thet0$Gamma)   #- a vector of length J Thet0$Gamma #
  Thet$BetaZ <- Thet0$BetaZ # rnorm(n = ncol(Z) +1)
  Thet$siggam <- sig02.del # 0.1
  Thet$tau0 <- tau0
  bnd <- GamBnd(tau0)
  # Thet$gamL <- min(bnd[1:2]); Thet$gamU <- max(bnd[1:2])
  Thet$gamL <- min(bnd[1])
  Thet$gamU <- max(bnd[2])

  # bs20 <- bs(a, df = 6, intercept = T)
  Thet$Betadel <- rep(1, ncol(Mn)) # c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))

  Thet$Cik.x <- sample.int(Kx, size = n, replace = T) # Thet0$Cik.x #sample(x = 1:Kx,size = n, replace = T)    #- vector of length Kx
  Thet$Cik.e <- sample.int(Ke, size = n, replace = T) # Thet0$Cik.e #sample(x = 1:Ke,size = n, replace = T)     #- vector of length Ku
  Thet$Cik.w <- sample.int(Kw, size = n, replace = T) # Thet0$Cik.w  #sample(x = 1:Kw,size = n, replace = T)     #- vector of length Kw
  Thet$Cik.m <- sample.int(Km, size = n, replace = T) # Thet0$Cik.m #sample(x = 1:Km,size = n, replace = T)     #- vector of length Km


  Thet$Pik.x <- rdirch(n = 1, alpha = rep(alpha.X, Kx) / Kx) # Thet0$Pik.x #rdirch(n=1,alpha = rep(alpha.X, Kx)/Kx)   #- vector of length Kx
  Thet$Pik.e <- rdirch(n = 1, alpha = rep(alpha.e, Ke) / Ke) # Thet0$Pik.e #rdirch(n=1,alpha = rep(alpha.e, Ke)/Ke)  #- vector of length Ku
  Thet$Pik.w <- rdirch(n = 1, alpha = rep(alpha.e, Kw) / Kw) # Thet0$Pik.w #rdirch(n=1,alpha = rep(alpha.W, Kw)/Kw)  #- vector of length Kw
  Thet$Pik.m <- rdirch(n = 1, alpha = rep(alpha.m, Km) / Km) # Thet0$Pik.m #rdirch(n=1,alpha = rep(alpha.m, Km)/Km)  #- vector of length Kw

  # Thet$pk.x   #- vector of length Kx
  # Thet$pk.u   #- vector of length Ku
  # Thet$pk.w   #- vector of length Kw

  Thet$sig2k.e <- nimble::rinvgamma(n = Ke, shape = 5 / 2, scale = 8 / 2) # Thet0$sig2k.e  #rgamma(n=Ke, shape=1,scale=.1) # matrix dim Kx x J
  Thet$gamk <- runif(n = Ke, min = .95 * Thet$gamL, max = .95 * Thet$gamU)
  Thet$p <- 1 * (Thet$gamk < 0) + (Thet$tau0 - 1 * (Thet$gamk < 0)) / (2 * pnorm(-abs(Thet$gamk)) * exp(0.5 * Thet$gamk * Thet$gamk))
  Thet$Ak <- (1 - 2 * Thet$p) / (Thet$p * (1 - Thet$p))
  Thet$Bk <- 2 / (Thet$p * (1 - Thet$p))
  Thet$Ck <- 1 / (1 * (Thet$gamk > 0) - Thet$p)
  Thet$si <- truncnorm::rtruncnorm(n = n, a = 0, mean = 0, sd = sqrt(Thet$sig2k.e[1]))
  Thet$nui <- rexp(n = n, rate = 1 / Thet$sig2k.e[1])
  # Thet$muk.e  =  Thet0$muk.e #rnorm(n=Ke)  #- Matrix of (Ku x J) and Ku x J - Note that $\Omega$ is a diagonal matrix

  Thet$Sigk.w <- repmat(c(.5 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$InvSigk.w <- repmat(c(2 * diag(pn)), n = Kw, m = 1) #- vector of length Kw
  Thet$Muk.w <- matrix(rnorm(n = Kw * pn), nrow = Kw) #- vectors of length Kw and Kw respectively

  Thet$Sigk.x <- repmat(c(.5 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$InvSigk.x <- repmat(c(2 * diag(pn)), Kx, 1) #-- vector of length Kx
  Thet$Muk.x <- matrix(rnorm(n = Kx * pn), nrow = Kx) #- vectors of length Kx and Kx respectively

  Thet$Sigk.m <- repmat(c(.5 * diag(pn)), Km, 1) #-- vector of length Kx
  Thet$InvSigk.m <- repmat(c(2 * diag(pn)), Km, 1)
  Thet$Muk.m <- matrix(rnorm(n = Km * pn), nrow = Km) #- vectors of length Kx and Kx respectively

  A <- min(c(c(Wn), c(Mn))) - .5 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .5 * diff(range(c(Wn, Mn)))
  # Thet$X <- rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  # matrix(rnorm(n = n*pn), ncol=pn)
  # Thet$X <- tmvtnorm::rtmvnorm(n=n,mean=rep(0,pn), sigma=diag(pn),lower = rep(A, pn),upper = rep(B, pn), algorithm = "gibbs")
  Thet$X <- .5 * (Mn + Wn) # Xn #TruncatedNormal::rtmvnorm(n=n,mu=rep(0,pn), sigma=diag(pn),lb = rep(A, pn), ub = rep(B, pn))


  # Input
  # Y (n x J)
  # Z (n x M)
  cat("\n#---- Loading updating functions ---> \n")
  #---------------------------------------------
  cat("# Update things related to Y --->\n")
  #--------------------------------------------
  # gal


  # method = "L-BFGS-B"
  # method = "BFGS"

  # Res = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), Thet=Thet, hessian = T, lower=c(.05,0.95*Thet$gamL), upper=c(30, 0.95*Thet$gamU))
  # Res

  # Res$hessian
  # Res$hessian/n
  # solve(Res$hessian/n)

  # LogPlostGalLik(c(.05,), c(ei), Thet)

  #-------- Update Beta (vector of length pn)
  #--- Prior parms
  #--- Sig0.gam0 = diag(pn); Mu0.gam0 = numeric(pn)
  # updateBetaZ <- function(Thet, Y, Z){
  #
  #   Zmod <- Z#cbind(1,Z)
  #   Y.td <- Y - Thet$muk.e[Thet$Cik.e] - Thet$X%*%Thet$Gamma
  #
  #   #InvSig0.gam0 <- as.inverse(Sig0.gam0)
  #   X_sc <- t(scale(t(Zmod), center=F, scale=Thet$sig2k.e[Thet$Cik.e])) ## n*pn
  #
  #   Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc)))
  #   Mu.gam <- Sig.gam%*%crossprod(X_sc, Y.td) #+ InvSig0.gam0%*%Mu0.gam0)
  #
  #   c(mvtnorm::rmvnorm(n=1,mean=c(Mu.gam),sigma = as.positive.definite(Sig.gam)))
  #
  # }
  # updateBetaZ <- cmpfun(updateBetaZ)

  #--------------------------------------------------------------
  #-------- Update Gamma (vector of length pn + the Z together)
  #--------------------------------------------------------------
  #--- Prior parms
  pn0 <- ncol(Z) + 1
  Pgam <- P.mat(pn) * tauval
  Sig0.gam0 <- diag(pn0)
  Mu0.gam0 <- numeric(pn0 + pn)
  # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))

  updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0) {
    # Zmod <-  Z #cbind(1,Z)
    Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
    InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
    Xmod <- cbind(Zmod, Thet$X)
    X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

    Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
    Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

    # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
    c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15, length(Mu.gam)), ub = rep(15, length(Mu.gam))))
  }
  updateGam <- cmpfun(updateGam)

  # siggam ~ IG(al0, bet0)
  al0 <- 0.001
  bet0 <- .005
  al0 <- 2.5978252 # 1
  bet0 <- 0.4842963 # .005
  # updatesigGam

  # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
  # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
  # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
  #------- Update Pik.e
  # alpha.e = 1.0
  # updates


  #--- update jointly sig2ke and gamke
  #-------------------------------------------------------------------------
  #-- output sig2k,gamk, p,accept=1, A, B, C
  # a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3
  a.sig <- 5 / 2
  b.sig <- 8 / 2
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 2.153640 1.167331
  a.sig <- 2.153640
  b.sig <- 1.167331
  sdsig.prop <- 0.2
  sdgam.prop <- .3 # 7.133445 4.303263
  a.sig <- 7.133445
  b.sig <- 4.303263
  sdsig.prop <- 0.2
  sdgam.prop <- .3 #
  # updates

  Sdsig.prop <- solve(-Thet0$Hess)
  diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
  cat("Dim", dim(Sdsig.prop))
  varxi <- c(10, 10, 10) * 1
  varxi <- c(10, 10, 10) * .25
  # varxi <- sqrt(c(5,5,5))
  # updates
  # updateCik.w <- Vectorize(updateCik.w,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * diag(pn)
  inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  Mu0.w <- rep(0, pn)
  Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
  inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  nu0.w <- pn + 2
  Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
  # updates
  # updateCik.m <- Vectorize(updateCik.m,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  # Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #
  Mu0.m <- rep(0, pn)
  Sig0.m <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.m <- as.inverse(Sig0.m) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates
  #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
  nu0.m <- pn + 2
  Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)

  # updates

  # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
  # prior values
  mu0.del <- 1.0 # sig02.del = 100
  # updates


  # prior values
  mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
  Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix

  # updates
  # updateCik.x <- Vectorize(updateCik.x,"i")
  #--- Update muk.w and sig2k.w based on the mixture of normal prior
  Mu0.x <- numeric(pn)
  Sig0.x <- .5 * diag(pn)
  inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
  Mu0.x <- rep(0, pn)
  Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
  inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
  # updates

  nu0.x <- pn + 2
  Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
  nu0.x <- pn + 2
  Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))
  # updates
  #
  # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
  # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

  A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
  B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))

  # updates

  #--- Write Rcpp function to speed things up
  {
    #
  }



  # Ot <- udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B)

  ## ----------------------------------------------------
  cat("\n #---- Create containers for the MCMC output ---> \n")
  #----------------------------------------
  #----- Divide these in sections - Construct the output
  #----- containers
  #--------------------------------------------------
  # Nsim <- 5000
  # Nsim = 5000
  MCMCSample <- list()

  #-- for X
  MCMCSample$X <- matrix(0, nrow = Nsim, ncol = length(c(Thet$X))) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$Muk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn)
  MCMCSample$Sigk.x <- matrix(0, nrow = Nsim, ncol = Kx * pn * pn)
  MCMCSample$Pik.x <- matrix(0, nrow = Nsim, ncol = Kx)
  MCMCSample$Cik.x <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for W
  MCMCSample$Muk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn)
  MCMCSample$Sigk.w <- matrix(0, nrow = Nsim, ncol = Kw * pn * pn)
  MCMCSample$Pik.w <- matrix(0, nrow = Nsim, ncol = Kw)
  MCMCSample$Cik.w <- matrix(0, nrow = Nsim, ncol = length(Y))

  #-- for M
  MCMCSample$Muk.m <- matrix(0, nrow = Nsim, ncol = Km * pn)
  MCMCSample$Sigk.m <- matrix(0, nrow = Nsim, ncol = Km * pn * pn)
  MCMCSample$Pik.m <- matrix(0, nrow = Nsim, ncol = Km)
  MCMCSample$Cik.m <- matrix(0, nrow = Nsim, ncol = length(Y))
  MCMCSample$Delta <- numeric(length(Thet$Delta))
  MCMCSample$Betdel <- matrix(0, nrow = Nsim, ncol = length(Thet$Betadel))

  # For Y
  MCMCSample$gamk <- matrix(0, nrow = Nsim, ncol = Ke) ## c(t()) : how we transform the matrix (row combined)
  MCMCSample$sig2k.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Pik.e <- matrix(0, nrow = Nsim, ncol = Ke)
  MCMCSample$Cik.e <- matrix(0, nrow = Nsim, ncol = length(Y))

  MCMCSample$Gamma <- matrix(0, nrow = Nsim, ncol = length(Thet0$Gamma))
  MCMCSample$BetaZ <- matrix(0, nrow = Nsim, ncol = length(Thet$BetaZ))
  MCMCSample$siggam <- numeric(Nsim)

  # MCMCSample$AcceptBetaS <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaS))
  #
  # MCMCSample$BetaX <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaX))
  #
  # MCMCSample$BetaZ <- matrix(0,nrow = Nsim, ncol=length(c(Thet$BetaZ))) ## c(t()) : how we transform the matrix (row combined)
  #
  # MCMCSample$BetaV <- matrix(0,nrow = Nsim, ncol=length(Thet$BetaV))
  # MCMCSample$AcceptBetaV <- numeric(Nsim)
  # Accept <-
  #
  MCMCSample$AcceptX <- matrix(0, nrow = Nsim, ncol = Ke)
  # MCMCSample$X <- matrix(0,nrow = Nsim, ncol=nrow(Y))
  # stop()
  #------------------------------------------------------------------
  # Nsim = 5000
  # Nsim = 1000
  # Nsim = 3000
  # library(profvis)
  # profvis(
  # system.time(
  # cat("\n #--- MCMC is starting ----> \n")

  for (iter in 1:Nsim) {
    if (iter %% 500 == 0) {
      cat("iter=", iter, "\n")
    }
    # #----------------------------------------------
    # # Update things related to X
    # #--------------------------------------------
    # #--- update Pik.m for epsilon_i
    # Thet$Pik.x <- updatePik.x(Thet, alpha.x)
    #
    # #--- update Cik.m for epsilon_i based on mixture of normals
    # Thet$Cik.x <- mapply(updateCik.x, i = 1:n, MoreArgs = list(Thet=Thet))
    # #Thet$Cik.x <- F2Cmp(CikX(Thet$X, Thet$Muk.x, Thet$Sigk.x, Thet$Pik.x))
    #
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.x <- updateMuk.x(Thet, Mu0.x, inSig0.x, Sig0.x)
    #
    # Thet$Sigk.x <- updateSigk.x(Thet, nu0.x, Psi0.x)
    # Thet$InvSigk.x <- InvCov(Thet$Sigk.x)
    #
    # #Ytd = Y - Zmod%*%Thet$BetaZ
    # #if(iter%%10 == 0){
    # #Thet$X <- udapateXi2(Thet, Y, Zmod, Wn, Mn, A=A, B=B) # Xn #
    # #Thet$X <- t(udapateXi2V(1:length(Y),Thet, Y, Zmod, Wn, Mn, A=A, B=B))
    # #Thet$X <- Xn #udapateXi2V(Thet, Y, Zmod, Wn, Mn, A=A, B=B)
    # #Thet$X <- UpdateX(Yd = Y, Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w, Sigm = Thet$Sigk.m, Sigx = Thet$Sigk.x, Cikw = Thet$Cik.w,
    # #                   Cikm = Thet$Cik.m, Cikx = Thet$Cik.x, A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x)
    #
    # Thet$X <- UpdateX(Wd = Wn - Thet$Muk.w[Thet$Cik.w,], Md = Mn - Thet$Muk.m[Thet$Cik.m,], Sigw = Thet$InvSigk.w[Thet$Cik.w,], Sigm = Thet$InvSigk.m[Thet$Cik.m,], Sigx = Thet$InvSigk.x[Thet$Cik.x,], A = rep(A, ncol(Wn)), B = rep(B, ncol(Wn)), Mukx = Thet$Muk.x[Thet$Cik.x,])
    #
    # #}
    # #dim(t(mapply(udapateXi, 1:length(Y), MoreArgs = list(Thet=Thet, Y = Y, W=Wn, M=Mn, A=A, B=B))))
    #
    # #Yd = (Y - Thet$muk.e[Thet$Cik.e])#/Thet$sig2k.e[Thet$Cik.e]
    # #Wd = Wn - Thet$Muk.w[Thet$Cik.w,]
    # #Md = (Mn - Thet$Muk.m[Thet$Cik.m,])*Thet$Delta
    #
    # #Thet$X <-
    # #  dim(UpdateX(Yd,Wd,Md,Thet$Muk.x, Thet$sig2k.e, Thet$Sigk.w,Sigm = Thet$Sigk.m, Thet$Sigk.x, Thet$Cik.e, Thet$Cik.w,
    # #                  Thet$Cik.m, Thet$Cik.x, Thet$Gamma, Thet$Delta, rep(A,J), rep(B,J)))
    # #---------------------------------------------
    # #--- update things related to W
    # #---------------------------------------------
    # #--- Update Pik.w
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Kw)
    # for(k in 1:Kw){ nk[k] = sum(Thet$Cik.w == k)}
    # #update the Pis
    # alpha.wnew <- alpha.W/Kw + nk
    # Thet$Pik.w <- rdirch(n=1, alpha.wnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.w <- mapply(updateCik.w, i= 1:N, MoreArgs = list(Thet=Thet, W = Wn))
    # Thet$Cik.w <- F2Cmp(CikW(Wn, Thet$X, Thet$Muk.w, Thet$Sigk.w, Thet$Pik.w))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.w <- updateMuk.w(Thet, Wn, Mu0.w, inSig0.w, Sig0.w )
    # Thet$Sigk.w <- updateSigk.w(Thet, Wn, nu0.w, Psi0.w)
    # Thet$InvSigk.w <- InvCov(Thet$Sigk.w)
    # #---------------------------------------------
    # #--- update things related to M
    # #---------------------------------------------
    # #--- Update Pik.w
    # #updatePik.w <- function(Thet, alpha.W){
    # nk <- numeric(Km)
    # for(k in 1:nrow(Thet$Muk.m)){ nk[k] = sum(Thet$Cik.m == k)}
    # #update the Pis
    # alpha.mnew <- alpha.m/Kw + nk
    # Thet$Pik.m <- rdirch(n=1, alpha.mnew)
    #
    # #--- Update Ci.k mixture of normals
    # #Thet$Cik.m <- mapply(updateCik.m, i= 1:n, MoreArgs = list(Thet=Thet, M = Mn))
    # Thet$Cik.m <- F2Cmp(CikMvec(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, Thet$Betadel))
    # #Thet$Cik.m <- F2Cmp(CikM(Mn, Thet$X, Thet$Muk.m, Thet$Sigk.m, Thet$Pik.m, 1.))
    # #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Thet$Muk.m <- updateMuk.m(Thet, Mn, Mu0.m, inSig0.m, Sig0.m)
    # Thet$Sigk.m <- updateSigk.m(Thet, Mn, nu0.m, Psi0.m)
    # Thet$InvSigk.m <- InvCov(Thet$Sigk.m)
    # #if(iter%%50 == 0){
    # Thet$Betadel <- rep(1,ncol(Mn)) #updateFunDelta(Thet, Mn, mu0.del, Sig0.del)
    # #Thet$Betadel <- updateFunDeltaSt(Thet, Mn, mu0.del, Sig0.del) # Meth 2
    # #DeltComp(i, Xst, X)   DeltVec #c(updateFunDelta(Thet, Mn, mu0.del, Sig0.del))
    # #Thet$Delta <-  1.0 #updateDelta(Thet, Mn, mu0.del, sig02.del)
    # MCMCSample$Betdel[iter,] <- Thet$Betadel
    # #}
    #---------------------------------------------
    # cat("#--- update things related to Y or epsilon ")
    #---------------------------------------------
    #--- update Pik.e for epsilon_i
    nk <- numeric(length(Thet$sig2k.e))
    for (k in 1:length(Thet$sig2k.e)) {
      nk[k] <- sum(Thet$Cik.e == k)
    }
    # update the Pis
    alpha.enew <- alpha.e / length(Thet$sig2k.e) + nk
    Thet$Pik.e <- c(rdirch(n = 1, alpha.enew))

    # update the Cik.e
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    # Thet$Cik.e <- mapply(updateCik.e, i = 1:N, MoreArgs = list(Thet=Thet, ei = ei))
    Thet$Cik.e <- updateCik.e(1:length(Y), Thet, ei) # rep(1,n) #
    # Thet$Cik.e <- F2Cmp(Cike(c(Y), Thet$X, Thet$muk.e, Thet$sig2k.e, Thet$Pik.e,Thet$Gamma))
    # update nui  and si
    # Thet$muk.e <- updatemuk.e(Thet, Y, Z, mu0.e, sig20.e)
    Thet$nui <- updatenui(Thet, Y, Zmod)
    Thet$si <- updatesi(Thet, Y, Zmod)
    # nuisi <- updatenuisi(Thet, Y, Zmod)
    # Thet$nui <- nuisi[,1]; Thet$si <- nuisi[,2];
    # update Sigk.e
    # Thet$sig2k.e <- updatesig2k.e(Thet,Y, Z, gam0.e, sig20.esig)
    # update sigk and gamk
    #--- update jointly sig2ke and gamke
    # Temp <- updatesig_gam(1:Ke, Thet, Y, Zmod, a.sig=2, b.sig = 1., sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop)
    Temp <- updatesig_gamHes(1:Ke, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi)
    Thet$sig2k.e <- Temp[1, ]
    Thet$gamk <- Temp[2, ]
    Thet$p <- Temp[3, ]
    MCMCSample$AcceptX[iter, ] <- Temp[4, ]
    Thet$Ak <- Temp[5, ]
    Thet$Bk <- Temp[6, ]
    Thet$Ck <- Temp[7, ]

    # update Gamma
    # Thet$siggam <- sig02.del
    Tp0 <- updateGam(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0)
    Thet$BetaZ <- Tp0[c(1:ncol(Zmod))] # c(0, Tp0[c(2:ncol(Zmod))])  #Tp0[c(1:ncol(Zmod))]
    Thet$Gamma <- Tp0[-c(1:ncol(Zmod))]
    # Thet$siggam <- 0.1# sig02.del #0.05 #updateSigGam(Thet$Gamma, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #
    Thet$siggam <- sig02.del # updateSigGam(Thet, al0, bet0, order = 2)# updateSigGam <- function(Bet,al0, bet0, order = 2) ##1/tauval #sig02.del# sig02.del#

    # Thet$Gamma <- updateGam(Thet, Y, Sig0.gam0, Mu0.gam0)
    # #--- update etai
    # Thet$etai <- mapply(updateetai,i=1:n, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ))
    #
    # #--- update beta_x
    # #-- Prior parameters
    # Thet$BetaX <- c(updateBetaX(Thet, Y, Z, BsZ, Beta.x0, SigmaX.0))
    #
    # #--- upodate beta_{z,m}
    # #--  Prior
    # Thet$BetaZ <-  t(mapply(updateBetaZ, m = 1:M, MoreArgs = list(Thet = Thet, Y = Y, Z = Z, BsZ = BsZ, Betazm.0, SigmaDel.0)))
    #
    # #--- update X_i
    # # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , SpreadX=SpreadX))
    # Res <- mapply(updateXi, i=1:n, MoreArgs = list(Thet=Thet, Y = Y, W= W, Z= Z, KnotsX=KnotsX, KnotsXV=KnotsXV , A = minXval , B = maxXval ))
    # Thet$X <- c(Res[1,])
    # AcceptPropX <- c(Res[2,]); #mean(AcceptPropX)
    #
    # #--- update BetaS
    # Res <- mapply(updateBetaS, j = 1:J, MoreArgs = list(Thet= Thet, Y = Y, BetaX0 = BetaX0, sig2prop = sig2prop))
    # Thet$BetaS <- Res[1,]
    # AcceptPropBetaS <- c(Res[2,]);
    #
    # #--- update BetaV
    # Res <- updateBetaV(Thet, Y, BetaV0, Sig2prop, Sigprior)
    # Thet$BetaV <- Res$Val
    # AcceptPropBetaV <- Res$Pb

    #---------------------------------------------
    #--- Save output
    #-- For X
    MCMCSample$Muk.x[iter, ] <- c(t(Thet$Muk.x)) ## Stor rowwise
    MCMCSample$Sigk.x[iter, ] <- c(t(Thet$Sigk.x))
    MCMCSample$Pik.x[iter, ] <- Thet$Pik.x
    MCMCSample$Cik.x[iter, ] <- Thet$Cik.x
    MCMCSample$X[iter, ] <- c(t(Thet$X)) ## Store row-wise

    #-- for W
    MCMCSample$Muk.w[iter, ] <- c(t(Thet$Muk.w))
    MCMCSample$Sigk.w[iter, ] <- c(t(Thet$Sigk.w))
    MCMCSample$Pik.w[iter, ] <- Thet$Pik.w
    MCMCSample$Cik.w[iter, ] <- Thet$Cik.w

    #-- for M
    MCMCSample$Muk.m[iter, ] <- c(t(Thet$Muk.m))
    MCMCSample$Sigk.m[iter, ] <- c(t(Thet$Sigk.m))
    MCMCSample$Pik.m[iter, ] <- Thet$Pik.m
    MCMCSample$Cik.m[iter, ] <- Thet$Cik.m
    # MCMCSample$Delta[iter] <- Thet$Delta
    #-- For Y
    MCMCSample$gamk[iter, ] <- Thet$gamk ## c(t()) : how we transform the matrix (row combined)
    MCMCSample$sig2k.e[iter, ] <- c(Thet$sig2k.e)
    MCMCSample$Pik.e[iter, ] <- Thet$Pik.e
    MCMCSample$Cik.e[iter, ] <- Thet$Cik.e

    MCMCSample$Gamma[iter, ] <- Thet$Gamma
    MCMCSample$BetaZ[iter, ] <- Thet$BetaZ
    MCMCSample$siggam[iter] <- Thet$siggam
  }

  # )
  # )

  #---- Return the MCMC smaples
  cat("\n ---- MCMCs are done!! ---/n")
  MCMCSample$Delthat <- Delthat
  MCMCSample$Hess <- Sdsig.prop
  MCMCSample
}






  # Basis


  # p.mat

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

  # Basis


  # p.mat

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


  
