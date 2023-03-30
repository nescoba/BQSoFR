updateGam <- function(Thet, Y, Zmod, Sig0.gam0, Mu0.gam0) {
  # Zmod <-  Z #cbind(1,Z)
  Y.td <- Y - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si - Thet$Ak[Thet$Cik.e] * Thet$nui # Thet$muk.e[Thet$Cik.e]
  InvSig0.gam0 <- 1.0 * as.matrix(bdiag(as.inverse(Sig0.gam0), P.mat(length(Thet$Gamma)) / Thet$siggam))
  Xmod <- cbind(Zmod, Thet$X)
  X_sc <- sweep(Xmod, MARGIN = 1, (sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$Bk[Thet$Cik.e] * Thet$nui), FUN = "/") ## n*pn

  Sig.gam <- as.inverse(as.symmetric.matrix(crossprod(X_sc, Xmod) + InvSig0.gam0))
  Mu.gam <- c(Sig.gam %*% (crossprod(X_sc, Y.td) + InvSig0.gam0 %*% Mu0.gam0))

  # c(mvtnorm::rmvnorm(n=1, mean = Mu.gam, sigma = as.positive.definite(Sig.gam)))
  c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-6, length(Mu.gam)), ub = rep(6, length(Mu.gam))))

  # c(TruncatedNormal::rtmvnorm(n=1, mu = Mu.gam, sigma = as.positive.definite(Sig.gam), lb = rep(-15,length(Mu.gam)), ub = rep(15,length(Mu.gam))))
}

updateSigGam <- function(Thet, al0, bet0, order = 2) {
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

# hist(1/rgamma(n=50000,shape=al0,rate=bet0))
# hist(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
# mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0))
# mean(nimble::rinvgamma(n=50000,shape=al0,scale=bet0) < 1)
#------- Update Pik.e
# alpha.e = 1.0
updatePik.e <- function(Thet, alpha.e) {
  nk <- numeric(length(Thet$Cik.e))
  for (i in 1:length(Thet$Cik.e)) {
    nk[i] <- sum(Thet$Cik.e == i)
  }

  # update the Pis
  alpha.enew <- alpha.e / length(Thet$Cik.e) + nk

  rdirichlet(n = 1, alpha.enew)
}

#-------- Update Cik.e

# ei <- Y - Z%*%Thet$Betaz - Thet$X%*%Thet$Gamma
updateCik.e <- function(i, Thet, ei) {
  # Y - Thet$C*abs(Thet$gam)*Thet$si - Thet$A*Thet$nui

  #  Yi.tld <- Y[i] - c(crossprod(Thet$Gamma, Thet$X[i,]))
  # pi.k  <- Thet$Pik.e*mapply(dnorm, x = Yi.tld, mean=Thet$muk.e, sd = sqrt(Thet$sig2k.e)) + .Machine$double.eps
  pik <- Thet$Pik.e * mapply(fGal.p0, e = ei[i], sig2 = sqrt(Thet$sig2k.e), gam = Thet$gamk, p = Thet$p, tau0 = Thet$tau0) #+ .Machine$double.eps
  which(rmultinom(n = 1, size = 1, prob = pik) == 1)
}
updateCik.e <- Vectorize(updateCik.e, "i")

#--- update Si
#-- calss id for individual i (Matrix with columns of nui and si)
updatenui <- function(Thet, Y, Z) {
  ki <- Thet$Cik.e
  bi <- (((Y - Z %*% Thet$BetaZ - Thet$X %*% Thet$Gamma - Thet$Ck[ki] * abs(Thet$gamk[ki]) * Thet$si)^2) / (Thet$Bk[ki] * sqrt(Thet$sig2k.e[ki])))
  ai <- ((Thet$Ak[ki] * Thet$Ak[ki]) / (Thet$Bk[ki] * sqrt(Thet$sig2k.e[ki])) + 2 / sqrt(Thet$sig2k.e[ki]))

  # mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
  mapply(ghyp::rgig, n = 1, lambda = 1 / 2, chi = bi, psi = ai)
  # val = mapply(ghyp::rgig, n =1, lambda = 1/2, chi = bi, psi = ai) + 0.01
  # ifelse(val < .001, rexp(n = length(muui), rate = 1/sqrt(Thet$sig2k.e[Thet$Cik.e] )), val)
}

updatesi <- function(Thet, Y, Z) {
  #-- update si
  ki <- Thet$Cik.e
  sig2si <- 1 / (1 / Thet$sig2k.e[ki] + ((Thet$Ck[ki] * Thet$gamk[ki])^2) / (Thet$Bk[ki] * sqrt(Thet$sig2k.e[ki]) * Thet$nui))
  muui <- sig2si * (Thet$Ck[ki] * abs(Thet$gamk[ki]) * (Y - Z %*% Thet$BetaZ - Thet$X %*% Thet$Gamma - Thet$Ak[ki] * Thet$nui)) / (Thet$Bk[ki] * sqrt(Thet$sig2k.e[ki]) * Thet$nui)

  # mapply(truncnorm::rtruncnorm, n=1, a = 0.01, mean = muui, sd = sqrt(sig2si))
  mapply(truncnorm::rtruncnorm, n = 1, a = 0.001, mean = muui, sd = sqrt(sig2si))
}


#--- update jointly sig2ke and gamke
#-------------------------------------------------------------------------
#-- output sig2k,gamk, p,accept=1, A, B, C
# a.sig=4; b.sig = .25; sdsig.prop = 0.2; sdgam.prop = .3

updatesig_gam <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, sdsig.prop = sdsig.prop, sdgam.prop = sdgam.prop) {
  a0 <- 0.001
  idk <- which(Thet$Cik.e == k)
  if (length(idk) > 0) {
    ei <- Y - Zmod %*% Thet$BetaZ - Thet$X %*% Thet$Gamma
    sig_newk <- truncnorm::rtruncnorm(n = 1, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0)
    gam_newk <- truncnorm::rtruncnorm(n = 1, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95 * Thet$gamL, b = .95 * Thet$gamU)
    pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))

    logLkold <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sqrt(Thet$sig2k.e[k]), gam = Thet$gamk[k], p = Thet$p[k], tau0 = Thet$tau0) + .Machine$double.eps))
    + dgamma(sqrt(Thet$sig2k.e[k]), shape = a.sig, scale = b.sig, log = T) + dunif(Thet$gamk[k], min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
      - log(truncnorm::dtruncnorm(x = sqrt(Thet$sig2k.e[k]), mean = sig_newk, sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(Thet$gamk[k], mean = gam_newk, sd = sdgam.prop, a = 0.95 * Thet$gamL, b = .95 * Thet$gamU) + .Machine$double.eps))

    logLknew <- (sum(log(mapply(fGal.p0, e = ei[idk], sig2 = sig_newk, gam = gam_newk, p = pnew, tau0 = Thet$tau0) + .Machine$double.eps))
    + dgamma(sig_newk, shape = a.sig, scale = b.sig, log = T) + dunif(gam_newk, min = 0.95 * Thet$gamL, max = 0.95 * Thet$gamU, log = T)
      - log(truncnorm::dtruncnorm(x = sig_newk, mean = sqrt(Thet$sig2k.e[k]), sd = sdsig.prop, a = a0) + .Machine$double.eps) - log(truncnorm::dtruncnorm(gam_newk, mean = Thet$gamk[k], sd = sdgam.prop, a = 0.95 * Thet$gamL, b = .95 * Thet$gamU) + .Machine$double.eps))

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
      sig_newk <- rgamma(n = 1, shape = a.sig, scale = b.sig)
      gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU)
      pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
      Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
    }
  } else {
    sig_newk <- rgamma(n = 1, shape = a.sig, scale = b.sig)
    gam_newk <- runif(n = 1, min = 0.95 * Thet$gamL, max = .95 * Thet$gamU) # runif(n = 1, min = Thet$gamL , max = Thet$gamU)
    pnew <- 1 * (gam_newk < 0) + (Thet$tau0 - 1 * (gam_newk < 0)) / (2 * pnorm(-abs(gam_newk)) * exp(0.5 * gam_newk * gam_newk))
    Res <- c(sig_newk^2, gam_newk, pnew, 0, (1 - 2 * pnew) / (pnew * (1 - pnew)), 2 / (pnew * (1 - pnew)), 1 / (1 * (gam_newk > 0) - pnew))
  }

  Res
}
updatesig_gam <- Vectorize(updatesig_gam, "k")

updatesig_gamHes <- function(k, Thet, Y, Zmod, a.sig = a.sig, b.sig = b.sig, Sdsig.prop = Sdsig.prop, sdgam.prop = sdgam.prop, varxi = varxi) {
  a0 <- 0.1
  ab <- 5
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

updateMuk.w <- function(Thet, W, Mu0.w, inSig0.w, Sig0.w) {
  Ku <- length(Thet$Pik.w)
  J <- ncol(Thet$X)
  Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
  SigK <- list() #-- store the covariances
  SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
  Yi.tld <- W - Thet$X
  # Muk <- matrix(0, ncol=J,nrow=Ku)
  SigK <- list()
  for (k in 1:Ku) {
    idk <- which(Thet$Cik.w == k)
    nk <- length(idk)
    if (length(idk) > 0) {
      Yi0 <- matrix(c(Yi.tld[idk, ]), nrow = nk, byrow = F)
      TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[k, ], ncol = J, byrow = F)))
      SigK[[k]] <- as.inverse(as.symmetric.matrix(nk * TpSi + inSig0.w)) ## sum of matrices inverse
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

updateSigk.w <- function(Thet, W, nu0.w, Psi0.w) {
  Ku <- nrow(Thet$Muk.w)
  m <- ncol(W)
  al <- numeric(Ku)
  bet <- numeric(Ku)
  Wtl <- W - Thet$X - Thet$Muk.w[Thet$Cik.w, ]
  Res <- matrix(0, ncol = m * m, nrow = Ku)
  for (i in 1:Ku) {
    id <- which(Thet$Cik.w == i)
    nk <- length(id)
    if (nk > 0) {
      nun <- nu0.w + nk
      Yi0 <- matrix(c(Wtl[id, ]), nrow = nk, byrow = F)
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
updatePik.m <- function(Thet, alpha.m) {
  nk <- numeric(nrow(Thet$Muk.m))
  for (k in 1:nrow(Thet$Muk.m)) {
    nk[k] <- sum(Thet$Cik.m == k)
  }

  # update the Pis
  alpha.mnew <- alpha.m / nrow(Thet$Muk.m) + nk

  c(rdirichlet(n = 1, alpha.mnew))
}

#--- update Cik.m for epsilon_i based on mixture of normals

updateCik.m <- function(i, Thet, M) { # mixture of normals
  # id <- Thet$Cik.m[i]
  # Sig <- matrix(Thet$Sigk.m[id,], ncol = ncol(M), byrow = F)
  Km <- nrow(Thet$Muk.m)
  xi <- M[i, ] - Thet$Delta * Thet$X[i, ]
  # pi.k  = Thet$Pik.m*mapply(dmvnormVec, k = 1:Km, MoreArgs = list(x = xi , Mean = Thet$Muk.m , SigmaVc = Thet$Sigk.m))
  pi.k <- Thet$Pik.m * dmvnormVec(x = xi, Mean = Thet$Muk.m, SigmaVc = Thet$Sigk.m) + .Machine$double.eps
  which(rmultinom(n = 1, size = 1, prob = pi.k) == 1)
}
# updateCik.m <- Vectorize(updateCik.m,"i")
#--- Update muk.w and sig2k.w based on the mixture of normal prior
# Mu0.m = rep(0,pn) ; Sig0.m = .5*diag(pn); inSig0.m = 2*diag(pn) #as.inverse(.25*cov(Wn)) #diag(pn) #

updateMuk.m <- function(Thet, M, Mu0.m, inSig0.m, Sig0.m) {
  # cat(dim(inSig0.m),"(--1)\n")
  Ku <- length(Thet$Pik.m)
  J <- ncol(M)
  Mu.k <- matrix(0, nrow = Ku, ncol = J) # K x J
  SigK <- list() #-- store the covariances
  SigKWg <- matrix(0, ncol = J, nrow = J) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
  Yi.tld <- M - Thet$X %*% diag(Thet$Betadel)
  Muk <- matrix(0, ncol = J, nrow = Ku)
  for (k in 1:Ku) {
    idk <- which(Thet$Cik.m == k)
    nk <- length(idk)
    if (length(idk) > 0) {
      Yi0 <- matrix(c(Yi.tld[idk, ]), nrow = nk, byrow = F)
      TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[k, ], ncol = J, byrow = F)))
      SigK[[k]] <- as.inverse(as.symmetric.matrix(nk * TpSi + inSig0.m)) ## sum of matrices inverse
      # cat(dim(SigK[[k]]),"(--2 \n")
      SigKWg <- SigKWg + (Thet$Pik.m[k] * Thet$Pik.m[k]) * SigK[[k]] # J x J
      Mu.k[k, ] <- SigK[[k]] %*% (crossprod(TpSi, c(colSums(Yi0))) + inSig0.m %*% Mu0.m) ## some of mat
      # cat(dim(SigKWg),"(--3 \n")
    } else {
      SigK[[k]] <- Sig0.m
      SigKWg <- SigKWg + (Thet$Pik.m[k] * Thet$Pik.m[k]) * SigK[[k]] # J x J
      Mu.k[k, ] <- Mu0.m
    }
  }
  #--- we impose some constrainsts on the mean
  SIg0 <- bdiag(SigK) # create a block diagonal matrix
  SIGR0 <- NULL
  for (k in 1:Ku) {
    SIGR0 <- rbind(SIGR0, Thet$Pik.m[k] * SigK[[k]])
  } ## K*J x J
  MUR0 <- colSums(diag(Thet$Pik.m) %*% Mu.k) # return a evector of length J

  MURFin <- c(t(Mu.k)) - SIGR0 %*% solve(SigKWg) %*% MUR0 ## J*K
  SIGR0fin <- as.matrix(nearPD(SIg0 - SIGR0 %*% solve(SigKWg) %*% t(SIGR0), doSym = T)$mat) ## J*K x J*K

  #--- Simulate K-1 vectors from the degenerate dist.
  id <- 1:(J * (Ku - 1))
  SimMU.k <- matrix(c(mvtnorm::rmvnorm(n = 1, mean = MURFin[id], SIGR0fin[id, id])), ncol = J, byrow = T) ## (K-1) x J
  SimMU.k1 <- -colSums(diag(Thet$Pik.m[-Ku]) %*% SimMU.k) / Thet$Pik.m[Ku] ## get a K-1 x J matrix follow by colSums - a vector of J
  as.matrix(rbind(SimMU.k, SimMU.k1)) ## k * J
}
updateMuk.m <- cmpfun(updateMuk.m)

updateSigk.m <- function(Thet, M, nu0.m, Psi0.m) {
  Ku <- nrow(Thet$Muk.m)
  m <- ncol(M)
  al <- numeric(Ku)
  bet <- numeric(Ku)
  Mtl <- M - Thet$X %*% diag(Thet$Betadel) - Thet$Muk.m[Thet$Cik.m, ]
  Res <- matrix(0, ncol = m * m, nrow = Ku)
  for (i in 1:Ku) {
    id <- which(Thet$Cik.m == i)
    nk <- length(id)
    if (nk > 0) {
      nun <- nu0.m + nk
      Yi0 <- matrix(c(Mtl[id, ]), nrow = nk, byrow = F)
      Psin.m <- as.positive.definite(Psi0.m + crossprod(Yi0))
      Res[i, ] <- c(rinvwishart(nun, Psin.m))
    } else {
      Res[i, ] <- c(rinvwishart(nu0.m, Psi0.m))
    }
  }
  Res
}
updateSigk.m <- cmpfun(updateSigk.m)

#---- Update things related to delta
quadFormVec <- function(i, X, sigma, Val) {
  id <- Val[i]
  as.symmetric.matrix(quad.form(as.inverse(as.symmetric.matrix(matrix(sigma[id, ], ncol = length(X[i, ]), byrow = F))), diag(c(X[i, ]))))
}
quadFormVec <- Vectorize(quadFormVec, "i", SIMPLIFY = F) ## Resulting in a list if precision matrices
quadFormVec <- cmpfun(quadFormVec)

quadFormVec3 <- function(i, Mi, X, sigma, Val) {
  id <- Val[i]
  c(quad.3form.inv(matrix(sigma[id, ], ncol = length(X[i, ]), byrow = F), diag(X[i, ]), Mi[i, ]))
}
quadFormVec3 <- Vectorize(quadFormVec3, "i", SIMPLIFY = F)
quadFormVec3 <- cmpfun(quadFormVec3)

# SimDElVec(1:5, Mn - matrix(SinResVec$X[1,],ncol=8),
# prior values
updateDelta <- function(Thet, M, mu0.del, sig02.del) {
  n <- nrow(M)
  Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
  Res <- quadFormVec(i = 1:nrow(Thet$X), X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
  Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)
  sig20 <- 1 / (1 / sig02.del + sum(Res))
  mu <- sig20 * (sum(Res3) + mu0.del / sig02.del)
  rtruncnorm(n = 1, a = 0, b = 10, mean = mu, sd = sqrt(sig20))

  # c(sum(Res3), sum(Res),mu, sqrt(sig20), rtruncnorm(n=1, a=0, b=Inf, mean = mu, sd = sqrt(sig20)))
}
updateDelta <- cmpfun(updateDelta)


updateFunDelta <- function(Thet, M, mu0.del, Sig0.del) {
  Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
  Res <- quadFormVec(i = 1:nrow(Thet$X), X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
  Res3 <- quadFormVec3(i = 1:nrow(Thet$X), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)
  Sig20 <- as.inverse(Sig0.del + as.symmetric.matrix(Reduce("+", Res)))
  Mu <- Sig20 %*% (matrix(c(Reduce("+", Res3)), ncol = 1) + Sig0.del %*% mu0.del)
  c(TruncatedNormal::rtmvnorm(n = 1, mu = c(Mu), sigma = Sig20, lb = rep(0, length(c(Mu))))) # , ub = rep(B,J)))
}
updateFunDelta <- cmpfun(updateFunDelta)


SimDElVec <- function(i, Mi, X, Sigma, Val) {
  id <- Val[i]
  Ot <- mvtnorm::rmvnorm(n = 1, mean = Mi[i, ], sigma = as.symmetric.matrix(matrix(Sigma[id, ], ncol = length(X[i, ]), byrow = F)))
}
SimDElVec <- Vectorize(SimDElVec, "i")

#--
DeltComp <- function(k, Xst, X) {
  abs(sum(Xst[, k] * X[, k]) / (sum(X[, k]^2)))
  # abs(diag(as.inverse(crossprod(X))%*%crossprod(X,Xst)))
}
DeltComp <- Vectorize(DeltComp, "k")

updateFunDeltaSt <- function(Thet, M, mu0.del, Sig0.del) {
  Mitl <- M - Thet$Muk.m[Thet$Cik.m, ]
  Xst <- base::t(SimDElVec(1:nrow(M), Mitl, Thet$X, Thet$Sigk.m, Thet$Cik.m)) ## K by n matrix and now I get a n by K
  c(DeltComp(1:ncol(Thet$X), Xst, Thet$X))
  # c(DeltComp(Xst, Thet$X))

  # Res <- quadFormVec(i = 1:nrow(Thet$X),X = Thet$X, sigma = Thet$Sigk.m, Val = Thet$Cik.m)
  # Res3 <- quadFormVec3(i = 1:nrow(Thet$X),Mitl, Thet$X, Thet$Sigk.m,Thet$Cik.m)
  # Sig20 <- as.inverse( Sig0.del + as.symmetric.matrix(Reduce('+', Res)))
  # Mu <- Sig20%*%(matrix(c(Reduce('+',Res3)), ncol=1) + Sig0.del %*% mu0.del)
  # c(TruncatedNormal::rtmvnorm(n=1, mu = c(Mu), sigma =  Sig20,lb = rep(0,length(c(Mu)))))  #, ub = rep(B,J)))
}
updateFunDeltaSt <- cmpfun(updateFunDeltaSt)


#----------------------------------------------
# Update things related to X
#--------------------------------------------------
#--- update Pik.m for epsilon_i
# alpha.x <- 1.0
updatePik.x <- function(Thet, alpha.x) {
  nk <- numeric(nrow(Thet$Muk.x))
  for (k in 1:nrow(Thet$Muk.x)) {
    nk[k] <- sum(Thet$Cik.x == k)
  }

  # update the Pis
  alpha.xnew <- alpha.x / nrow(Thet$Muk.x) + nk

  c(rdirichlet(n = 1, alpha.xnew))
}

#--- update Cik.m for epsilon_i based on mixture of normals
updateCik.x <- function(i, Thet) { # mixture of normals
  # Kx = ncol(Thet$X) # J x J
  # id <- Thet$Cik.x[i]
  # Sig <- matrix(Thet$Sigk.x[id,], ncol = length(Thet$Gamma), byrow = F)
  Xi <- Thet$X[i, ]
  # pi.k  = Thet$Pik.x*mapply(dmvnormVec, k = 1:Kx, x = Xi,  Mean = Thet$Muk.x , SigmaVc = Thet$Sigk.x)
  pi.k <- Thet$Pik.x * dmvnormVec(x = Xi, Mean = Thet$Muk.x, SigmaVc = Thet$Sigk.x) + .Machine$double.eps
  which(rmultinom(n = 1, size = 1, prob = pi.k) == 1)
}

updateMuk.x <- function(Thet, Mu0.x, inSig0.x, Sig0.x) {
  K <- length(Thet$Pik.x)
  J <- ncol(Thet$X)
  Res0 <- matrix(0, nrow = K, ncol = J)
  Mu.k <- matrix(0, nrow = K, ncol = J) # K x J
  # SigKWg <- matrix(0, ncol = ncol(Y),nrow=ncol(Y)) #-- store the weights some of the covarianace - cov of \sum_{k=1}\pi_kMu_k
  Yi.tld <- Thet$X
  Muk <- matrix(0, ncol = J, nrow = K)
  for (k in 1:K) {
    idk <- which(Thet$Cik.x == k)
    nk <- length(idk)
    if (length(idk) > 0) {
      Yi0 <- matrix(c(Yi.tld[idk, ]), nrow = nk, byrow = F)
      TpSi <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[k, ], ncol = J, byrow = F)))
      SigK <- as.inverse(as.symmetric.matrix((nk * TpSi + inSig0.x))) ## sum of matrices inverse
      Muk <- SigK %*% (crossprod(TpSi, c(colSums(Yi0))) + inSig0.x %*% Mu0.x) ## some of mat
      Mu.k[k, ] <- c(mvtnorm::rmvnorm(n = 1, mean = Muk, sigma = SigK))
    } else {
      Mu.k[k, ] <- c(mvtnorm::rmvnorm(n = 1, mean = Mu0.x, sigma = Sig0.x))
    }
  }
  #--- we impose some constrainsts on the mean
  Mu.k
}
updateMuk.x <- cmpfun(updateMuk.x)

updateSigk.x <- function(Thet, nu0.x, Psi0.x) {
  Ku <- nrow(Thet$Muk.x)
  m <- ncol(Thet$Muk.x)
  Xtl <- Thet$X - Thet$Muk.x[Thet$Cik.x, ]
  Res <- matrix(0, ncol = m * m, nrow = Ku)
  for (i in 1:Ku) {
    id <- which(Thet$Cik.x == i)
    nk <- length(id)
    if (nk > 0) {
      nux <- nu0.x + nk
      Yi0 <- matrix(c(Xtl[id, ]), nrow = nk, byrow = F)
      Psin.x <- Psi0.x + crossprod(Yi0)
      Res[i, ] <- c(rinvwishart(nux, Psin.x))
    } else {
      Res[i, ] <- c(rinvwishart(nu0.x, Psi0.x))
    }
  }
  Res
}
updateSigk.x <- cmpfun(updateSigk.x)
#
# A <- min(c(Wn, Mn)) - .2*diff(range(c(Wn, Mn)))
# B <- max(c(Wn, Mn)) + .2*diff(range(c(Wn, Mn)))

udapateXi <- function(i, Thet, Y, W, M, Z, A = A, B = B) {
  ie <- Thet$Cik.e[i]
  iw <- Thet$Cik.w[i]
  im <- Thet$Cik.m[i]
  ix <- Thet$Cik.x[i]
  J <- ncol(Thet$X)

  Zmod <- Z # cbind(1,Z)
  Ytl <- (Y - Zmod %*% Thet$BetaZ - Thet$muk.e[Thet$Cik.e]) / Thet$sig2k.e[Thet$Cik.e]
  Wtl <- W - Thet$Muk.w[Thet$Cik.w, ]
  Mtl <- (M - Thet$Muk.m[Thet$Cik.m, ]) * Thet$Delta

  Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw, ], ncol = J, byrow = F)))
  Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im, ], ncol = J, byrow = F)))
  Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix, ], ncol = J, byrow = F)))

  # Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
  Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(tcrossprod(c(Thet$Gamma)) / Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2) * Sigm + Sigx)))

  Mu <- c(Sig %*% (c(Thet$Gamma) * Ytl[i] + crossprod(Sigw, Wtl[i, ]) + crossprod(Sigm, Mtl[i, ]) + crossprod(Sigx, Thet$Muk.x[ix, ])))

  # c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
  c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu, sigma = Sig, lb = rep(A, J), ub = rep(B, J)))
}
# udapateXi <- Vectorize(udapateXi,"i")
udapateXi <- cmpfun(udapateXi)


udapateXi2A <- function(Thet, Y, Zmod, Wn, Mn, A = A, B = B) {
  J <- ncol(Thet$X)
  n <- length(Y)
  # Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
  Ytl <- c((Y - Zmod %*% Thet$BetaZ - Thet$Ak[Thet$Cik.e] * Thet$nui - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si) / (Thet$Bk[Thet$Cik.e] * sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$nui)) # Thet$Gam
  Wtl <- Wn - Thet$Muk.w[Thet$Cik.w, ]
  Mtl <- Mn - Thet$Muk.m[Thet$Cik.m, ] # %*%diag(Thet$Betadel)
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

    Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw, ], ncol = J, byrow = F)))
    Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im, ], ncol = J, byrow = F)))
    Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix, ], ncol = J, byrow = F)))

    # Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat / (Thet$Bk[ie] * sqrt(Thet$sig2k.e[ie]) * Thet$nui[i]) + Sigw + Sigm + Sigx))) # quad.form(Sigm, diag(Thet$Betadel))

    Mu <- c(Sig %*% (c(Thet$Gamma) * Ytl[i] + crossprod(Sigw, Wtl[i, ]) + crossprod(Sigm, Mtl[i, ]) + crossprod(Sigx, Thet$Muk.x[ix, ])))

    # c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    # Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
    Res[i, ] <- c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu, sigma = Sig, lb = rep(A, J), ub = rep(B, J)))
  }
  Res
}
# udapateXi <- Vectorize(udapateXi,"i")
udapateXi2A <- cmpfun(udapateXi2A)

## This approach of simulating Xi only uses W, M, and X

udapateXi2 <- function(Thet, Y, Zmod, Wn, Mn, A = A, B = B) {
  J <- ncol(Thet$X)
  n <- length(Y)
  # Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
  Ytl <- c((Y - Zmod %*% Thet$BetaZ - Thet$Ak[Thet$Cik.e] * Thet$nui - Thet$Ck[Thet$Cik.e] * abs(Thet$gamk[Thet$Cik.e]) * Thet$si) / (Thet$Bk[Thet$Cik.e] * sqrt(Thet$sig2k.e[Thet$Cik.e]) * Thet$nui)) # Thet$Gam
  Wtl <- Wn - Thet$Muk.w[Thet$Cik.w, ]
  Mtl <- Mn - Thet$Muk.m[Thet$Cik.m, ] # %*%diag(Thet$Betadel)
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
    Sigm <- matrix(Thet$InvSigk.m[im, ], ncol = J, byrow = F)
    Sigx <- matrix(Thet$InvSigk.x[ix, ], ncol = J, byrow = F)

    # Thet$InvSigk.m

    # Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat / (Thet$Bk[ie] * sqrt(Thet$sig2k.e[ie]) * Thet$nui[i]) + Sigw + Sigm + Sigx))) # quad.form(Sigm, diag(Thet$Betadel))
    # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel))

    Mu <- c(Sig %*% (c(Thet$Gamma) * Ytl[i] + crossprod(Sigw, Wtl[i, ]) + crossprod(Sigm, Mtl[i, ]) + crossprod(Sigx, Thet$Muk.x[ix, ])))

    # Mu <- c(Sig%*%(crossprod(Sigw,Wtl[i,]) + crossprod(Sigm,Mtl[i,]) + crossprod(Sigx, Thet$Muk.x[ix,])));

    # c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    # Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
    Res[i, ] <- c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu, sigma = Sig, lb = rep(A, J), ub = rep(B, J)))
  }
  Res
}
# udapateXi <- Vectorize(udapateXi,"i")
udapateXi2 <- cmpfun(udapateXi2)


udapateXi2V <- function(Thet, Y, Zmod, Wn, Mn, A = A, B = B) {
  J <- ncol(Thet$X)
  n <- length(Y)
  # Ytl = (Y - Thet$muk.e[Thet$Cik.e])/Thet$sig2k.e[Thet$Cik.e]
  # Ytl <- c((Y - Zmod%*%Thet$BetaZ - Thet$Ak[Thet$Cik.e]*Thet$nui - Thet$Ck[Thet$Cik.e]*abs(Thet$gamk[Thet$Cik.e])*Thet$si)/(Thet$Bk[Thet$Cik.e]*sqrt(Thet$sig2k.e[Thet$Cik.e])*Thet$nui)) #Thet$Gam
  Wtl <- Wn - Thet$Muk.w[Thet$Cik.w, ]
  Mtl <- Mn - Thet$Muk.m[Thet$Cik.m, ] # %*%diag(Thet$Betadel)
  Res <- matrix(0, ncol = J, nrow = n)
  Sigw <- matrix(0, J, J)
  Sigm <- matrix(0, J, J)
  Sigx <- matrix(0, J, J)
  # GamMat= tcrossprod(c(Thet$Gamma))

  for (i in 1:n) {
    ie <- Thet$Cik.e[i]
    iw <- Thet$Cik.w[i]
    im <- Thet$Cik.m[i]
    ix <- Thet$Cik.x[i]

    # Sigw <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.w[iw,],ncol=J, byrow=F)))
    # Sigm <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.m[im,],ncol=J, byrow=F)))
    # Sigx <- as.inverse(as.symmetric.matrix(matrix(Thet$Sigk.x[ix,],ncol=J, byrow=F)))#

    Sigw <- matrix(Thet$InvSigk.w[iw, ], ncol = J, byrow = F)
    Sigm <- matrix(Thet$InvSigk.m[im, ], ncol = J, byrow = F)
    Sigx <- matrix(Thet$InvSigk.x[ix, ], ncol = J, byrow = F)

    # Thet$InvSigk.m

    # Sig <- 1.0*as.matrix(nearPD(solve(tcrossprod(c(Thet$Gamma))/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx))$mat)
    # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/Thet$sig2k.e[ie] + Sigw + (Thet$Delta^2)*Sigm + Sigx)))
    # Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(GamMat/(Thet$Bk[ie]*sqrt(Thet$sig2k.e[ie])*Thet$nui[i]) + Sigw + Sigm + Sigx))) #quad.form(Sigm, diag(Thet$Betadel))
    Sig <- as.positive.definite(as.inverse(as.symmetric.matrix(Sigw + Sigm + Sigx))) # quad.form(Sigm, diag(Thet$Betadel))

    Mu <- (Sig %*% (crossprod(Sigw, Wtl[i, ]) + crossprod(Sigm, Mtl[i, ]) + crossprod(Sigx, Thet$Muk.x[ix, ])))

    # c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J),algorithm = "gibbs",thinning=10))
    # Res[i,] =c(tmvtnorm::rtmvnorm(n=1, mean = Mu, sigma =  Sig,lower=rep(A,J), upper=rep(B,J)))
    Res[i, ] <- c(TruncatedNormal::rtmvnorm(n = 1, mu = Mu, sigma = Sig, lb = rep(A, J), ub = rep(B, J)))
  }
  Res
}
# udapateXi2V <- Vectorize(udapateXi2V,"i")
udapateXi2V <- cmpfun(udapateXi2V)
