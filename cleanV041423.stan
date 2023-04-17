//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

functions {
  
    /*
    A = -est*p_neg + .5*pow(gam, 2)*pow(p_neg/p_pos, 2) + log(Phi_approx(a2-a3)) + log1m_exp(fabs(log(Phi_approx(a2-a3)) - log(Phi_approx(a2))));
  gam = (gamU - gamL) * ligam + gamL;
  real gam = (gamU - gamL) * ligam + gamL;
  real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau, real gamL, real gamU){
  real GAL2_rng(real mu, real sigma, real ligam, real tau, real gamL, real gamU){
   */
     /* helper function for asym_laplace_lpdf
  * Args:
    *   y: the response value
  *   tau: quantile parameter in (0, 1)
  */
    real rho_quantile(real y, real tau) {
      if (y < 0) {
        return y * (tau - 1);
      } else {
        return y * tau;
      }
    }
  /* asymmetric laplace log-PDF for a single response
  * Args:
    *   y: the response value
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
      return log(tau * (1 - tau)) -
        log(sigma) -
        rho_quantile((y - mu) / sigma, tau);
    }
  /* asymmetric laplace log-CDF for a single quantile
  * Args:
    *   y: a quantile
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lcdf(real y, real mu, real sigma, real tau) {
      if (y < mu) {
        return log(tau) + (1 - tau) * (y - mu) / sigma;
      } else {
        return log1m((1 - tau) * exp(-tau * (y - mu) / sigma));
      }
    }
  /* asymmetric laplace log-CCDF for a single quantile
  * Args:
    *   y: a quantile
  *   mu: location parameter
  *   sigma: positive scale parameter
  *   tau: quantile parameter in (0, 1)
  * Returns:
    *   a scalar to be added to the log posterior
  */
    real asym_laplace_lccdf(real y, real mu, real sigma, real tau) {
      if (y < mu) {
        return log1m(tau * exp((1 - tau) * (y - mu) / sigma));
      } else {
        return log1m(tau) - tau * (y - mu) / sigma;
      }
    }

   real GAL2_lpdf(real y, real mu, real sigma, real ligam, real tau){

   real p_pos;
   real p_neg;
   real a3;
   real a2;
   real p;
   real est;
   real A;
   real B;
   real Res = 0;
   real gam = ligam;
    p = (gam < 0) + (tau - (gam < 0))/(2*Phi(-fabs(gam))*exp(.5*pow(gam, 2)));
    p_pos = p -  (gam > 0);
    p_neg = p -  (gam < 0);
    est = (y - mu) / sigma;

    if(fabs(gam) > 0){
    a3 = p_pos * (est / fabs(gam));
    a2 = fabs(gam) * (p_neg / p_pos);


    if(est/gam > 0){
      A =  0.5*pow(gam, 2)*pow(p_neg/p_pos, 2) - est*p_neg + log_diff_exp(log(Phi_approx(a2-a3)), log(Phi_approx(a2)) );
      B =  0.5*pow(gam, 2) - p_pos*est + log(Phi_approx(-fabs(gam) + a3));
      Res = log(2*p*(1-p)) - log(sigma) +  log_sum_exp(A, B);
    }else{
      Res =  log(2*p*(1-p)) - log(sigma) - p_pos * est + 0.5 * pow(gam, 2) + log(Phi_approx(-fabs(gam) ));
    }
    }else{
    Res = asym_laplace_lpdf( y | mu, sigma, tau);
    }

    return Res;
   }

  real GAL2_rng(real mu, real sigma, real ligam, real tau){

     real A;
     real B;
     real C;
     real p;
     real hi;
     real nui;
     real mui=0;
     real Up = uniform_rng(.5, 1.0);

     real gam = ligam;
     p = (gam < 0) + (tau - (gam < 0))/(2*Phi_approx(-fabs(gam))*exp(.5*pow(gam, 2)));
     A = (1 - 2*p)/(p - pow(p,2));
     B = 2/(p - pow(p,2));
     C = 1/((gam > 0) - p);

      hi = sigma * inv_Phi(Up);
     nui = sigma * exponential_rng(1);
     mui += mu + A * nui + C * fabs(gam) * hi;

     return normal_rng(mui, sqrt(sigma*B*nui));
  }
  
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  // data for splines
  int Ks;  // number of linear effects
  matrix[N, Ks] Xs;  // design matrix for the linear effects
  // data for spline s(x)
  int nb_1;  // number of bases
  int knots_1[nb_1];  // number of knots
  // basis function matrices
  matrix[N, knots_1[1]] Zs_1_1;
  // data for splines
  int Ks_sigma;  // number of linear effects
  matrix[N, Ks_sigma] Xs_sigma;  // design matrix for the linear effects
  // data for spline s(x)
  int nb_sigma_1;  // number of bases
  int knots_sigma_1[nb_sigma_1];  // number of knots
  // basis function matrices
  matrix[N, knots_sigma_1[1]] Zs_sigma_1_1;
  int prior_only;  // should the likelihood be ignored?
  
  // Below are the values you need to append to the data created by Brms
  int pn; // Number of basis function and the is the number of columns for Wn
  vector[2] Bd; // the boundary values for Bd
  matrix[N,pn] Wn; // the projected values of Wn
  real<lower=0,upper=1> tau0;
}
transformed data {
}
parameters {
  real Intercept;  // temporary intercept for centered predictors
  vector[Ks] bs;  // spline coefficients
  // parameters for spline s(x)
  // standarized spline coefficients
  // Prior specification for the Ar prior on the the coefficents for Wn
  vector[pn] bw;
  real<lower=0.01,upper=.99> psi;
  real<lower=0> teta;
  vector[knots_1[1]] zs_1_1;
  real<lower=0> sds_1_1;  // standard deviations of spline coefficients
  real Intercept_sigma;  // temporary intercept for centered predictors
  vector[Ks_sigma] bs_sigma;  // spline coefficients
  // parameters for spline s(x)
  // standarized spline coefficients
  vector[knots_sigma_1[1]] zs_sigma_1_1;
  real<lower=0> sds_sigma_1_1;  // standard deviations of spline coefficients
  real<lower=Bd[1],upper=Bd[2]> gam;
}
transformed parameters {
  // actual spline coefficients
  vector[knots_1[1]] s_1_1;
  // actual spline coefficients
  vector[knots_sigma_1[1]] s_sigma_1_1;
  real tau = 0.25;
  real lprior = 0;  // prior contributions to the log posterior
  //bwa ar(1) stationary prior
  vector[pn] bwa;
  //real tau = tau0;
        // compute bwa
  bwa[1] = bw[1];
  for(i in 2:pn)
  bwa[i] = psi*bwa[i-1] + bw[i]*teta;
  
  
  lprior += std_normal_lpdf(bw);
  lprior += uniform_lpdf(psi | 0.01, .99);
  lprior += uniform_lpdf(gam | Bd[1], Bd[2]);
  // compute actual spline coefficients
  s_1_1 = sds_1_1 * zs_1_1;
  // compute actual spline coefficients
  s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1;
  lprior += student_t_lpdf(Intercept | 3, 0.1, 2.7);
  lprior += student_t_lpdf(sds_1_1 | 3, 0, 2.7)
    - 1 * student_t_lccdf(0 | 3, 0, 2.7);
  lprior += student_t_lpdf(Intercept_sigma | 3, 0, 2.5);
  lprior += student_t_lpdf(sds_sigma_1_1 | 3, 0, 2.7)
    - 1 * student_t_lccdf(0 | 3, 0, 2.7);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] sigma = rep_vector(0.0, N);
    // note I have added the term Wn * bwa 
    mu += Intercept + Xs * bs + Zs_1_1 * s_1_1 + Wn * bwa;
    sigma += Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1 + Wn * bwa;
    sigma = exp(sigma);
    for (n in 1:N) {
      target += GAL2_lpdf(Y[n] | mu[n], sigma[n], gam, tau0);
    }
  }
  // priors including constants
  target += lprior;
  target += std_normal_lpdf(zs_1_1);
  target += std_normal_lpdf(zs_sigma_1_1);
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept;
  // actual population-level intercept
  real b_sigma_Intercept = Intercept_sigma;
 // I added this part so swe cna compare model performance  
 vector[N] mu = rep_vector(0.0, N);
 vector[N] sigma = rep_vector(0.0, N);
 vector[N] log_lik = rep_vector(0.0, N);
 
 mu += Intercept + Xs * bs + Zs_1_1 * s_1_1 + Wn * bwa;
sigma += Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1 + Wn * bwa;
    sigma = exp(sigma);
  for(n in 1:N)
    log_lik[n] +=  GAL2_lpdf(Y[n] | mu[n], sigma[n], gam, tau0);

}