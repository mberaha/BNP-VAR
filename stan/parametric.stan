data {
  int<lower=1> numPatients;
  int<lower=1> maxT;
  int tByPatient[numPatients];

  int<lower=1> rdim;
  int<lower=1> ldim;
  int<lower=1> fixedDim;

  real ys[numPatients, maxT, rdim];
  real xs[numPatients, maxT, ldim];
  real zs[numPatients, fixedDim];
}

parameters {
  real<lower=0.0001> omega;
  // cholesky_factor_corr[rdim] S_Omega_chol;
  // vector<lower=0>[rdim] S_tau;
  matrix[rdim, ldim] B;
  matrix[rdim, fixedDim] Gamma;
  matrix[rdim, rdim] Phi0;

  // vector[rdim*rdim] phi00;
  // cholesky_factor_corr[rdim*rdim] V0_Omega;
  // vector<lower=0>[rdim*rdim] V0_tau;
}

// transformed parameters {
//    corr_matrix[rdim] S_Omega = S_Omega_chol * S_Omega_chol';
// }

model {
  vector[rdim] fixedMean;
  vector[rdim] longMean;
  vector[rdim] arMean;
  vector[rdim] prevY;
  vector[rdim] currY;

  matrix[rdim, rdim] eye = diag_matrix(rep_vector(1, rdim));


  to_vector(B) ~ normal(0, 5);
  to_vector(Gamma) ~ normal(0, 5);
  to_vector(Phi0) ~ normal(0, 1);

  // S_Omega_chol ~ lkj_corr_cholesky(5);

  // S_tau ~ cauchy(0, 2.5);
  omega ~ uniform(0.0001, 5);

  for (i in 1:numPatients) {
    fixedMean = Gamma * to_vector(zs[i, ]);
    for (t in 2:tByPatient[i]) {
      longMean = B * to_vector(xs[i, t,]);
      arMean = Phi0 * to_vector(ys[i, t-1, ]);
      target += normal_lpdf(
        to_vector(ys[i, t,]) | arMean + longMean + fixedMean,
        omega);
    }
  }
}

// generated quantities {
  //  matrix[rdim, rdim] Sigma = quad_form_diag(S_Omega_chol * S_Omega_chol', S_tau);
// }

