   P.mat <- function(K){
    # penalty matrix
    D <- diag(rep(1,K))
    D <- diff(diff(D))
    P <- t(D)%*%D 
    return(P)
  }
  
  
  InitialValues <- function(Y, W, M, Z, df0=3, pn, Gx, a){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- Z #cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    lmFit <- lm(Y ~ -1 + Zmod + W_i) 
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- residuals.lm(lmFit)
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  
  
  
  InitialValuesQR <- function(Y, W, M, Z, df0=3, pn, Gx, a, tau0){
    Res <- list()
    #bs.w <- splines::bs(a, df = pn+1, degree = min(c(df0,pn-1)), intercept=TRUE) 
    Zmod <- cbind(1,Z)
    
    bs2 <- splines::bs(a, df = pn, degree = min(c(df0,pn-1)), intercept=TRUE) 
    
    Wn <- crossprod(t(W), bs2)/length(a)
    Mn <- crossprod(t(M), bs2)/length(a)
    
    WsmB <- Wn
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, W[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    X0 = model.matrix(~Z+Wn)
    lmFit <- quantreg::rq(Y ~ -1 + X0, tau=tau0) 
    #summary(lmFit)
    plot(a, bs2%*%lmFit$coefficients[-c(1:3)], type="l")
    
    c_hatW <- lmFit$coefficients
    Res$Gamma <- as.numeric(c_hatW[-c(1:ncol(Zmod))])
    Res$BetaZ <- as.numeric(c_hatW[c(1:ncol(Zmod))])
    Res$Delta <- abs(mean(Mn)/mean(W_i))
    Res$Betadel <- c(abs(solve(t(bs2)%*%bs2)%*%t(bs2)%*%(colMeans(M)/colMeans(W))))
    #beta_tW <- crossprod(t(bs3), c_hatW)  #(t,1) matrix
    #-----------------------------------------------------
    #--- determine the Hessian matrix
    #=====================================================
    method = "L-BFGS-B"
    ei <- lmFit$residuals
    bnd <- GamBnd(tau0)
    Tp0 = optim(c(.2,-.1), LogPlostGalLik, method = method, ei = c(ei), tau0 = tau0, control=list("fnscale"=-1),hessian = T, lower=c(.005,0.95*min(bnd)), upper=c(30, 0.95*max(bnd) ))
    Res$Hess <- Tp0$hessian
    
    #--- Determine the number of clusters for epsilon_i
    #-----
    reslmfit <- lmFit$residuals
    BIC <- mclustBIC(reslmfit)
    mod1 <- Mclust(reslmfit, x = BIC, G = 2)
    Temp <- summary(mod1, parameters = TRUE)
    Res$Ke <- Temp$G + Gx
    Res$muk.e <- numeric(Res$Ke)
    Res$muk.e[1:Temp$G] = Temp$mean; 
    Res$sig2k.e <- rep(1, Res$Ke)*sum(lmFit$residuals^2)/lmFit$df.residual
    Res$sig2k.e[1:Temp$G] = Temp$variance
    Res$Cik.e <- as.numeric(Temp$classification)
    Res$Pik.e <- numeric(Res$Ke)
    Res$Pik.e[1:Temp$G] <- Temp$pro 
    #-----------------------------------------------------
    #--- Determine the number of cluster (W)
    #------
    Ui <- Wn #- W_i 
    Mod1 <- Mclust(Ui, G = 2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kw <- Tp$G +Gx
    Res$Muk.w <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.w <- as.matrix( rbind(t(Tp$mean),  repmat(rep(0,pn), Gx, 1)))
    
    Res$Sigk.w <- matrix(0, nrow = Res$Kw, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.w[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kw){Res$Sigk.w[j,] <- c( as.positive.definite(.5*diag(pn))) }
    Res$Cik.w <- Tp$classification
    Res$Pik.w <- Tp$pro
    #-------------------------------------------------   
    #--- Determine the number of cluster (X)
    #-----
    WsmB <- (W + M) /(1+Res$Delta)
    WsM <- NULL
    for(i in 1:nrow(W)){ WsM <- rbind(WsM, smooth.spline(a, WsmB[i,])$y)}
    W_i <- crossprod(t(WsM),bs2)/length(a)   #(n,k) matrix
    Ui <- W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Kx <- Tp$G + Gx
    Res$Muk.x <- matrix(0, ncol = pn, nrow=Res$Kw)
    Res$Muk.x <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    
    Res$Sigk.x <- matrix(0, nrow = Res$Kx, ncol = pn*pn)
    for(j in 1:Tp$G){Res$Sigk.x[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Kx){Res$Sigk.x[j,] <- c(.5*diag(pn))}
    Res$Cik.x <- Tp$classification
    Res$Pik.x <- Tp$pro  
    Res$X <- W_i
    #---------------------------------------------------
    #---- Determine the number of cluster (M)
    #------
    
    Ui <- Mn  #- Res$Delta*W_i 
    Mod1 <- Mclust(Ui, G=2)
    Tp <- summary(Mod1, parameters=T)
    Res$Km <- Tp$G + Gx
    #Res$Muk.m <- as.matrix(t(Tp$mean))
    Res$Muk.m <- as.matrix( rbind(t(Tp$mean),  repmat(colMeans(Ui), Gx, 1 )))
    Res$Sigk.m <- matrix(0,nrow=Res$Km,ncol=pn*pn)
    for(j in 1:Tp$G){Res$Sigk.m[j,] <- c(Tp$variance[,,j])}
    for(j in (Tp$G+1):Res$Km){Res$Sigk.m[j,] <- c(as.positive.definite(.5*diag(pn)))}
    Res$Cik.m <- Tp$classification
    Res$Pik.m <- Tp$pro
    #------- return result 
    Res 
  }
  