
#-- Using replicated W not M
BQBayes.ME_SoFRFuncSimFxFSMI <- function(Y, W, M, Z, a, Nsim, sig02.del = 0.1, tauval = .1, Delt, alpx = 1, tau0 = 0.5, X, g0, tunprop) {
    met <- c("f1", "f2", "f3", "f4", "f5")




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
    # @ New parameters to use
    # @parms si, ui, p, tau0, sig2k.e[k]; gamk[k], gamL; gamU
    # @sig :  variance


    #--------------------------------------------------------------
    #-------- Update Gamma (vector of length pn + the Z together)
    #--------------------------------------------------------------
    #--- Prior parms
    pn0 <- ncol(Z) + 1
    Pgam <- P.mat(pn) * tauval
    Sig0.gam0 <- diag(pn0)
    Mu0.gam0 <- numeric(pn0 + pn)
    # InvSig0.gam0 <-  1.0*as.matrix(bdiag(as.inverse(Sig0.gam0), Pgam))


    # siggam ~ IG(al0, bet0)
    al0 <- 0.001
    bet0 <- .005
    al0 <- 2.5978252 # 1
    bet0 <- 0.4842963 # .005

    al0 <- 0.001 # 1 #1
    bet0 <- 0.001 # 0.005 #.005

    # hist(1/rgamma(n=50000,shape=al0,rate=bet0))
    # hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
    # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
    # mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
    #------- Update Pik.e
    # alpha.e = 1.0



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


    Sdsig.prop <- solve(-Thet0$Hess)
    diag(Sdsig.prop) <- abs(diag(Sdsig.prop))
    cat("Dim", dim(Sdsig.prop))
    varxi <- c(10, 10, 10) * 1
    # varxi <- c(10,10,10)*.25
    # varxi <- c(10,10,10)*.15 ## high acceptance rate
    varxi <- c(10, 10, 10) * .35 ## -- High acceptance rate
    varxi <- c(10, 10, 10) * tunprop ## --
    # varxi <- sqrt(c(5,5,5))

    # updateCik.w <- Vectorize(updateCik.w,"i")
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    # Mu0.w = rep(0, pn);  Sig0.w = .5*diag(pn);inSig0.w = 2*diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
    Mu0.w <- rep(0, pn)
    Sig0.w <- .5 * diag(pn)
    inSig0.w <- 2 * diag(pn) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #
    Mu0.w <- rep(0, pn)
    Sig0.w <- .5 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w), ncol = pn, nrow = pn))
    inSig0.w <- as.inverse(Sig0.w) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #


    # nu0.w = pn + 2; Psi0.w = Sig0.w = cov(Wn) #.5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)
    nu0.w <- pn + 2
    Psi0.w <- 0.5 * (nu0.w - pn - 1) * cov(Wn) # .5*diag(pn) # as.symmetric.matrix(.5*cov(Wn)) #diag(pn)


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


    #  nu0.m = pn + 2; Psi0.m = .5*diag(pn)#as.symmetric.matrix(.25*cov(Wn))  #diag(pn)
    nu0.m <- pn + 2
    Psi0.m <- .5 * (nu0.m - pn - 1) * as.symmetric.matrix(cov(Wn)) # diag(pn)



    # SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
    # prior values
    mu0.del <- 1.0 # sig02.del = 100



    # prior values
    mu0.del <- rep(0, length(Thet$Betadel)) # sig02.del = 20
    Sig0.del <- P.mat(length(Thet$Betadel)) * .1 # Thet$siggam) # Inverse or precision covariance matrix


    # updateCik.x <- Vectorize(updateCik.x,"i")
    #--- Update muk.w and sig2k.w based on the mixture of normal prior
    Mu0.x <- numeric(pn)
    Sig0.x <- .5 * diag(pn)
    inSig0.x <- 2 * diag(pn) # as.inverse(.25*cov(Wn)) #.5*diag(pn) #
    Mu0.x <- rep(0, pn)
    Sig0.x <- .25 * as.symmetric.matrix(matrix(colMeans(Thet0$Sigk.w) + colMeans(Thet0$Sigk.m), ncol = pn, nrow = pn))
    inSig0.x <- as.inverse(Sig0.x) # as.inverse(.5*cov(Wn)) #.5*diag(pn) #


    nu0.x <- pn + 2
    Psi0.x <- .5 * diag(pn) # as.symmetric.matrix(.25*cov(Wn))
    nu0.x <- pn + 2
    Psi0.x <- .5 * (nu0.x - pn - 1) * cov(.5 * (Wn + Mn)) # as.symmetric.matrix(.25*cov(Wn))

    # A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
    # B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

    A <- min(c(Wn, Mn)) - .1 * diff(range(c(Wn, Mn)))
    B <- max(c(Wn, Mn)) + .1 * diff(range(c(Wn, Mn)))




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
