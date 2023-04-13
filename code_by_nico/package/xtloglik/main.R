#---- This will fit the model using STAN
#-------------------------------------------------------------------------------
#-- Implementing Model in Rstan
library(brms)
library(rstan)
library(splines)

#-------------------------------------------------------------------------------
#--- Import Data
#-------------------------------------------------------------------------------
source("~/Library/CloudStorage/OneDrive-IndianaUniversity/Join_folder_Annie_DrZoh/NHANES Dataset/2013-2014/BQReg_Nhanes2012_13Application.R")
load("/Users/rszoh/Library/CloudStorage/OneDrive-IndianaUniversity/Join_folder_Annie_DrZoh/NHANES Dataset/2013-2014/FinalDataV2.RData")

#-------------------------------------------------------------------------------
#--- Set up the data
#-------------------------------------------------------------------------------
pn <- 10
a <- seq(0, 1, length = dim(Warray)[2])
bs2 <- bs(a, df = pn, intercept = T)


# Xt <- apply(Warray, c(1,2), mean)
Xt <- as.matrix(FUI(model = "gaussian", smooth = T, Warray, silent = FALSE)) # } #- (abs(range(W)[1]) +1)
# Xt <- as.matrix(FUI(model="gaussian",smooth=T, Warray,silent = FALSE))
Wt <- (Xt %*% bs2) / length(a)


#---- Transform the data
# df_sample$AgeYRTru <- df_sample$AgeYR
# #df_sample$AgeYR <- scale(df_sample$AgeYR, center = T, scale = T)
# df_sampleOld <- df_sample
# df_sample$AgeYR <- df_sample$AgeYR/100
# df_sample$MnASTP <- df_sample$MnASTP*10
# df_sample$MnMVPA <-  df_sample$MnMVPA/100


df_sampleWt <- data.frame(df_sample, W = Wt)[1:200, ]



#********************************************************************************
# trial code (Cases we need to look at)
#
#-------------------------------------------------------------------------------
# case1 (Fit the model in brms ignoring MVPA and ASTP)
# case2 (Fit the model in STAN with smoothing prior and ignoring MVPA and ASTP)
# case3 (Fit the model in brms with MVPA and ASTP)
# Case4 (Fit the model in STAN with MVPA and ASTP and smoothing prior)

#-------------------------------------------------------------------------------
#---- Data set prep.
#-------------------------------------------------------------------------------
tau0 <- .1

form <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df_sampleWt)[grepl("W", colnames(df_sampleWt))]), collapse = " + "))
Dt0 <- make_standata(bf(as.formula(form), quantile = tau0), data = df_sampleWt[c(1:nrow(df_sampleWt)), ], family = asym_laplace())
# , stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))


fitCase1 <- list()
fitCase1b <- list()
fitCase2 <- list()
fitCase2b <- list()
fitCase3 <- list()
fitCase4 <- list()
fitCase5 <- list()
tauVal <- c(.1, .5, .9)
Niter <- 100
Nchain <- 2


for (l in 1:length(tauVal)) {
    tau0 <- tauVal[l]
    Bd <- GamBnd(tau0)[1:2]

    Dt0$pn <- pn
    Dt0$Xold <- Dt0$X
    Dt0$X <- Dt0$Xold[, !grepl("W.[0-9]", colnames(Dt0$Xold))]
    Dt0$Wn <- Dt0$Xold[, grepl("W.[0-9]", colnames(Dt0$Xold))]
    dim(Dt0$Wn)
    Dt0$M <- numeric(Dt0$pn)
    Dt0$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn)))) * .1
    Dt0$K <- ncol(Dt0$X)
    Bd <- GamBnd(tau0)[1:2]
    Dt0$Bd <- Bd * .99
    Dt0$tau0 <- tau0

    varX <- c("MnASTP", "MnMVPA")

    #-------------------------------------------------------------------------------
    # Case 1 (no MPVA and ASTP)
    #-------------------------------------------------------------------------------
    # @' This function is the stan_funs2 for the GAL distribution

    Bd <- GamBnd(tau0)[1:2]

    # {
    GAL2 <- custom_family(
        "GAL2",
        dpars = c("mu", "sigma", "gam", "tau"), links = c("identity", "log", "identity", "identity"),
        lb = c(NA, 0, Bd[1] * .9, 0), ub = c(NA, NA, Bd[2] * .9, 1), type = "real"
    ) # , vars = "vint1[n]"

    stan_funs2 <- readLines("functions.stan")

    # }

    #--- Now define all of these here
    form <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df_sampleWt)[grepl("W", colnames(df_sampleWt))]), collapse = " + "))

    stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
    fitCase1[[l]] <- brms::brm(bf(as.formula(form), tau = tau0),
        data = df_sampleWt, family = GAL2, stanvars = stanvars2,
        chains = Nchain, iter = Niter, control = list(adapt_delta = 0.99), cores = 2, seed = 1123
    ) # init = 0.1,


    #-------------------------------------------------------------------------------
    # Case 1b (with MPVA and ASTP) - Wt
    #-------------------------------------------------------------------------------
    # @' This function is the stan_funs2 for the GAL distribution

    Bd <- GamBnd(tau0)[1:2]

    #   {
    GAL2 <- custom_family(
        "GAL2",
        dpars = c("mu", "sigma", "gam", "tau"), links = c("identity", "log", "identity", "identity"),
        lb = c(NA, 0, Bd[1] * .9, 0), ub = c(NA, NA, Bd[2] * .9, 1), type = "real"
    ) # , vars = "vint1[n]"

    stan_funs2 <- readLines("functions.stan")

    # }

    #--- Now define all of these here
    formCase1b <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR"), collapse = " + "))
    for (pa in varX) {
        formCase1b <- paste0(formCase1b, " + s(", pa, ", bs='ps', k=11)")
    }

    stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
    fitCase1b[[l]] <- brms::brm(bf(as.formula(formCase1b), tau = tau0),
        data = df_sampleWt,
        family = GAL2, stanvars = stanvars2,
        chains = Nchain, iter = Niter, control = list(adapt_delta = 0.99), cores = 2, seed = 1123
    ) # init = 0.1,

    StandCodeVGalLin <- readLines("StandCodeVGalLin.stan")

    df2 <- df_sampleWt

    form2 <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df2)[grepl("W", colnames(df2))]), collapse = " + "))
    Dt2 <- make_standata(bf(as.formula(form2), quantile = tau0), data = df2[c(1:nrow(df2)), ], family = asym_laplace())

    Dt2$pn <- pn
    Dt2$Xold <- Dt2$X
    Dt2$X <- Dt2$Xold[, !grepl("W.[0-9]", colnames(Dt2$Xold))]
    Dt2$Wn <- Dt2$Xold[, grepl("W.[0-9]", colnames(Dt2$Xold))]
    dim(Dt2$Wn)
    Dt2$M <- numeric(Dt2$pn)
    Dt2$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn)))) * .1
    Dt2$K <- ncol(Dt2$X)
    Bd <- GamBnd(tau0)[1:2]
    Dt2$Bd <- Bd * .9
    Dt2$tau0 <- tau0
    Dt2$X[, ncol(Dt2$X)] <- Dt2$X[, ncol(Dt2$X)] / 100


    fitCase2[[l]] <- stan(
        model_code = StandCodeVGalLin,
        data = Dt2, iter = Niter, chains = Nchain, verbose = TRUE, control = list(adapt_delta = 0.99, max_treedepth = 11), cores = 2, seed = 1123
    )

    #-------------------------------------------------------------------------------
    # Case 2b - Without MVPA and ASTP (smooth Wt)  (with mixture distribution)
    #-------------------------------------------------------------------------------
    {
        StandCodeVGalLin <- readLines("StandCodeVGalLin2.stan")
    }

    df2b <- df_sampleWt

    form1 <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df2b)[grepl("W", colnames(df2b))]), collapse = " + "))
    Dt2b <- make_standata(bf(as.formula(form1), quantile = tau0), data = df2b[c(1:nrow(df2b)), ], family = asym_laplace())

    Dt2b$pn <- pn
    Dt2b$Xold <- Dt2b$X
    Dt2b$X <- Dt2b$Xold[, !grepl("W.[0-9]", colnames(Dt2b$Xold))]
    Dt2b$Wn <- Dt2b$Xold[, grepl("W.[0-9]", colnames(Dt2b$Xold))]
    dim(Dt2b$Wn)
    Dt2b$M <- numeric(Dt2b$pn)
    Dt2b$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn)))) * .1
    Dt2b$K <- ncol(Dt2b$X)
    Bd <- GamBnd(tau0)[1:2]
    Dt2b$Bd <- Bd * .9
    Dt2b$tau0 <- tau0
    Dt2b$X[, ncol(Dt2b$X)] <- Dt2b$X[, ncol(Dt2b$X)] / 100
    Dt2b$Kep <- 2

    fitCase2b[[l]] <- stan(
        model_code = StandCodeVGalLin,
        data = Dt2b, iter = Niter, chains = Nchain, verbose = TRUE, control = list(adapt_delta = 0.99, max_treedepth = 11), cores = 2, seed = 1123
    )

    #-------------------------------------------------------------------------------
    # Case 3
    #-------------------------------------------------------------------------------
    formCase3 <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df_sampleWt)[grepl("W", colnames(df_sampleWt))]), collapse = " + "))
    for (pa in varX) {
        formCase3 <- paste0(formCase3, " + s(", pa, ", bs='ps', k=11)")
    }

    df_sampleWtSd <- df_sampleWt
    df_sampleWtSd$AgeYR <- df_sampleWtSd$AgeYR / sd(df_sampleWtSd$AgeYR)

    stanvars2 <- stanvar(scode = stan_funs2, block = "functions")
    fitCase3[[l]] <- brms::brm(bf(as.formula(formCase3), tau = tau0),
        data = df_sampleWt, family = GAL2,
        stanvars = stanvars2, chains = Nchain, iter = Niter, control = list(adapt_delta = 0.99), cores = 2, seed = 1123
    ) # init = 0.1,




    #-------------------------------------------------------------------------------
    # Case 4
    #-------------------------------------------------------------------------------
    df4 <- df_sample
    df4$AgeYRTru <- df4$AgeYR
    # df4$AgeYR <- scale(df4$AgeYR, center = T, scale = T)

    df4$AgeYR <- df_sample$AgeYR / 100
    df4$MnASTP <- df_sample$MnASTP * 6
    df4$MnMVPA <- df_sample$MnMVPA / 100

    df_sampleWt4 <- data.frame(df4, W = Wt)
    # tau0 = .1

    form4 <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df_sampleWt4)[grepl("W", colnames(df_sampleWt4))]), collapse = " + "))
    for (pa in varX) {
        form4 <- paste0(form4, " + s(", pa, ", bs='ps', k=11)")
    }


    Dt4 <- brms::make_standata(bf(as.formula(form4), quantile = tau0), data = df_sampleWt4, family = asym_laplace())
    # , stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))
    Bd <- GamBnd(tau0)[1:2]

    Dt4$pn <- pn
    Dt4$Xold <- Dt4$X
    Dt4$X <- Dt4$Xold[, !grepl("W.[0-9]", colnames(Dt4$Xold))]
    Dt4$Wn <- Dt4$Xold[, grepl("W.[0-9]", colnames(Dt4$Xold))]
    dim(Dt4$Wn)
    Dt4$M <- numeric(Dt4$pn)
    Dt4$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn)))) * .1
    Dt4$K <- ncol(Dt4$X)
    Bd <- GamBnd(tau0)[1:2]
    Dt4$Bd <- Bd * .9
    Dt4$tau0 <- tau0

    #-- Option 2 GAL
    # fitXt[[l]]
    {
        StandCodeVGal <- readLines("StandCodeVGal.stan")
    }

    fitCase4[[l]] <- stan(
        model_code = StandCodeVGal,
        data = Dt4, iter = Niter, chains = Nchain, verbose = TRUE, control = list(adapt_delta = 0.99, max_treedepth = 11), cores = 2, seed = 1123
    )



    #-------------------------------------------------------------------------------
    # Case 5
    #-------------------------------------------------------------------------------
    df5 <- df_sample
    df5$AgeYRTru <- df5$AgeYR
    # df5$AgeYR <- scale(df5$AgeYR, center = T, scale = T)

    df5$AgeYR <- df_sample$AgeYR / 100
    df5$MnASTP <- df_sample$MnASTP * 6
    df5$MnMVPA <- df_sample$MnMVPA / 100

    df_sampleWt5 <- data.frame(df5, W = Wt)
    # tau0 = .1

    form5 <- paste0("Y ~ ", paste(c("Gender", "Race", "HealthCondt2", "AgeYR", colnames(df_sampleWt5)[grepl("W", colnames(df_sampleWt4))]), collapse = " + "))
    for (pa in varX) {
        form5 <- paste0(form5, " + s(", pa, ", bs='ps', k=11)")
    }


    Dt5 <- brms::make_standata(bf(as.formula(form4), quantile = tau0), data = df_sampleWt5, family = asym_laplace())
    # , stanvars = stanvars2, prior = bprior,chains = 2,iter = 500, control = list(adapt_delta = 0.97), data2 = list(M = rep(0, 29), V = SigbPrio))
    Bd <- GamBnd(tau0)[1:2]

    Dt5$pn <- pn
    Dt5$Xold <- Dt5$X
    Dt5$X <- Dt5$Xold[, !grepl("W.[0-9]", colnames(Dt5$Xold))]
    Dt5$Wn <- Dt5$Xold[, grepl("W.[0-9]", colnames(Dt5$Xold))]
    dim(Dt5$Wn)
    Dt5$M <- numeric(Dt5$pn)
    Dt5$V <- as.positive.definite(as.inverse(P.mat(pn) + diag(rep(.01, pn)))) * .1
    Dt5$K <- ncol(Dt5$X)
    Bd <- GamBnd(tau0)[1:2]
    Dt5$Bd <- Bd * .9
    Dt5$tau0 <- tau0

    #-- Option 2 GAL
    # fitXt[[l]]
    {
        StandCodeVGal5 <- readLines("StandCodeVGal5.stan")
    }

    fitCase5[[l]] <- stan(
        model_code = StandCodeVGal5,
        data = Dt5, iter = Niter, chains = Nchain, verbose = TRUE, control = list(adapt_delta = 0.99, max_treedepth = 11), cores = 2, seed = 1123
    )
}
