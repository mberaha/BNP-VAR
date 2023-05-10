#include "base_sampler.hpp"

BaseSampler::BaseSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXi> is_missing,
          std::vector<MatrixXd> long_covs, std::vector<VectorXd> fixed_covs)
    : H(H),
      orig_data(data),
      is_missing(is_missing),
      long_covs(long_covs),
      fixed_covs(fixed_covs) {
    
  imputed_data = orig_data;
  rdim = orig_data[0].cols();
  ldim = long_covs[0].cols();
  fdim = fixed_covs[0].size();
  ndata = data.size();
  clus_allocs.resize(ndata);

  sum_time_steps = 0.0;
  for (int i = 0; i < ndata; i++) sum_time_steps += orig_data[i].rows();
  missing_data = true;

}

BaseSampler::BaseSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXd> long_covs,
        std::vector<VectorXd> fixed_covs) : H(H) {
  orig_data = data;
  imputed_data = data;
  this->long_covs = long_covs;
  this->fixed_covs = fixed_covs;
  rdim = orig_data[0].cols();
  ldim = long_covs[0].cols();
  fdim = fixed_covs[0].size();
  ndata = data.size();
  clus_allocs.resize(ndata);
  sum_time_steps = 0.0;
  for (int i = 0; i < ndata; i++) sum_time_steps += orig_data[i].rows();

} 


void BaseSampler::set_prior_sigma(Eigen::MatrixXd sigma0_, double nu_) {
  if (sigma0_.size() > 0) {
    if (sigma0_.rows() != rdim) {
      std::string err_msg =
          std::string("sigma0_ must have dimension ") + std::to_string(rdim);
      throw std::invalid_argument(err_msg);
    }
  } else {
    sigma0_ = Eigen::MatrixXd::Identity(rdim, rdim);
  }
  sigma_0 = sigma0_;
  nu = nu_;
  std::cout << "end" << std::endl;
}

void BaseSampler::set_prior_beta(Eigen::MatrixXd beta0, double var) {
  std::cout << "BaseSampler::::set_prior_beta()" << std::endl;
  if (beta0.size() > 0) {
    if (beta0.cols() != ldim || beta0.rows() != rdim) {
      std::string err_msg = std::string("beta0 must have dimension (") +
                            std::to_string(rdim) + "," + std::to_string(ldim) +
                            ")";
      throw std::invalid_argument(err_msg);
    }
  } else {
    beta0 = Eigen::MatrixXd::Zero(ldim, rdim);
  }
  mu_beta = Map<VectorXd>(beta0.data(), beta0.size());
  sigma_beta = Eigen::MatrixXd::Identity(mu_beta.size(), mu_beta.size()) * var;
  sigma_beta_inv =
      Eigen::MatrixXd::Identity(mu_beta.size(), mu_beta.size()) / var;
  std::cout << "end" << std::endl;
}

void BaseSampler::set_prior_gamma(Eigen::MatrixXd gamma0, double var) {
  std::cout << "BaseSampler::::set_prior_gamma()" << std::endl;
  if (gamma0.size() > 0) {
    if (gamma0.cols() != fdim) {
      std::string err_msg = std::string("gamma0 must have dimension (") +
                            std::to_string(rdim) + "," + std::to_string(fdim) +
                            ")";
      throw std::invalid_argument(err_msg);
    }
  } else {
    gamma0 = Eigen::MatrixXd::Zero(fdim, rdim);
  }
  mu_gamma = Map<VectorXd>(gamma0.data(), gamma0.size());
  sigma_gamma =
      Eigen::MatrixXd::Identity(mu_gamma.size(), mu_gamma.size()) * var;
  sigma_gamma_inv =
      Eigen::MatrixXd::Identity(mu_gamma.size(), mu_gamma.size()) / var;
  std::cout << "end" << std::endl;
}

void BaseSampler::initialize() {
  std::cout << "BaseSampler::initialize()" << std::endl;
  beta_vec = mu_beta;
  gamma_vec = mu_gamma;
  
  
  sigma = sigma_0 / (nu - rdim * rdim + 1);
  std::cout << "sigma \n" << sigma << std::endl;
  sigma_inv = PrecMat(stan::math::inverse_spd(sigma));
  weights = VectorXd::Ones(H).array() / H;

  for (int i = 0; i < ndata; i++) {
    clus_allocs[i] =
        stan::math::categorical_rng(weights, Rng::Instance().get()) - 1;
  }
  std::cout << "BaseSampler::initialize() end" << std::endl;
}

// void BaseSampler::step() {
//   iter++;
// //   sample_beta();
// //   sample_sigma();
//   std::cout << "sample_gamma"  << std::endl; 
//   sample_gamma();
//   std::cout << "sample_gamma done" << std::endl;
//   std::cout << "sample_allocs" << std::endl;
//   sample_allocs();
//   std::cout << "sample_allocs done" << std::endl;
// } 

void BaseSampler::sample_beta() {
  MatrixXd curr_data(sum_time_steps - ndata, rdim);
  MatrixXd curr_preds(sum_time_steps - ndata, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
  int row = 0;
  for (int i = 0; i < ndata; i++) {
    int l = imputed_data[i].rows();

    MatrixXd phi = get_phi(clus_allocs[i], fixed_covs[i]);    
    
    for (int j = 1; j < l; j++) {
      curr_data.row(row) = (imputed_data[i].row(j).transpose() -
                            phi * imputed_data[i].row(j - 1).transpose() -
                            gamma_mat * fixed_covs[i])
                               .transpose();

      curr_preds.row(row) = long_covs[i].row(j);
      row += 1;
    }
  }

  MatrixXd xtransx = curr_preds.transpose() * curr_preds;
  MatrixXd beta_hat_ = xtransx.ldlt().solve(curr_preds.transpose() * curr_data);
  VectorXd beta_hat = Map<VectorXd>(beta_hat_.data(), beta_hat_.size());

  MatrixXd helper = kroneckerProduct(sigma_inv.get_prec(), xtransx);
  MatrixXd temp = helper + sigma_beta_inv;
  MatrixXd sigma_beta_post = stan::math::inverse_spd(temp);

  VectorXd mu_beta_post =
      sigma_beta_post * (helper * beta_hat + sigma_beta_inv * mu_beta);

  beta_vec = stan::math::multi_normal_rng(mu_beta_post, sigma_beta_post,
                                          Rng::Instance().get());
}

void BaseSampler::sample_gamma() {
//   std::cout << "BaseSampler::sample_gamma()" << std::endl;
  MatrixXd curr_data(sum_time_steps - ndata, rdim);
  MatrixXd curr_preds(sum_time_steps - ndata, fdim);
  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  int row = 0;
  for (int i = 0; i < ndata; i++) {
    int l = imputed_data[i].rows();
    MatrixXd phi = get_phi(clus_allocs[i], fixed_covs[i]);
    for (int j = 1; j < l; j++) {
      curr_data.row(row) = (imputed_data[i].row(j).transpose() -
                            phi * imputed_data[i].row(j - 1).transpose() -
                            beta_mat * long_covs[i].row(j).transpose())
                               .transpose();
      curr_preds.row(row) = fixed_covs[i].transpose();
      row += 1;
    }
  }

  MatrixXd xtransx = curr_preds.transpose() * curr_preds;
  MatrixXd gamma_hat_ =
      xtransx.ldlt().solve(curr_preds.transpose() * curr_data);
  VectorXd gamma_hat = Map<VectorXd>(gamma_hat_.data(), gamma_hat_.size());

  MatrixXd helper = kroneckerProduct(sigma_inv.get_prec(), xtransx);
  MatrixXd prec = helper + sigma_gamma_inv;
  prec += prec.transpose().eval();
  prec /= 2.0;

  VectorXd mu_gamma_post =
      prec.ldlt().solve(helper * gamma_hat + sigma_gamma_inv * mu_gamma);

  gamma_vec = stan::math::multi_normal_prec_rng(mu_gamma_post, prec,
                                                Rng::Instance().get());
//   std::cout << "BaseSampler::sample_gamma() end" << std::endl;
}

void BaseSampler::sample_sigma() {
    // std::cout << "BaseSampler::sample_sigma()" << std::endl;
  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

  double nu_post = nu + sum_time_steps - ndata;
  MatrixXd sigma_0_post = sigma_0;
  for (int i = 0; i < ndata; i++) {
    MatrixXd phi = get_phi(clus_allocs[i], fixed_covs[i]);
    for (int j = 1; j < imputed_data[i].rows(); j++) {
      VectorXd mean = phi * imputed_data[i].row(j - 1).transpose() +
                      beta_mat * long_covs[i].row(j).transpose() +
                      gamma_mat * fixed_covs[i];
      VectorXd datum = imputed_data[i].row(j).transpose();

      //   std::cout << "temp: " << temp.transpose() << std::endl;
      sigma_0_post += (datum - mean) * (datum - mean).transpose();
    }
  }

  sigma =
      stan::math::inv_wishart_rng(nu_post, sigma_0_post, Rng::Instance().get());
  sigma_inv = PrecMat(stan::math::inverse_spd(sigma));

//   std::cout << "sample sigma done" << std::endl;
}

void BaseSampler::sample_allocs() {
//   std::cout << "BaseSampler::sample_allocs()" << std::endl;
  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

  #pragma omp parallel for
  for (int i = 0; i < ndata; i++) {
    VectorXd probas = get_weights(fixed_covs[i], true, true);
    for (int h = 0; h < H; h++) {
      MatrixXd phi = get_phi(h, fixed_covs[i]);
      for (int j = 1; j < imputed_data[i].rows(); j++) {
        VectorXd currmean = (phi * imputed_data[i].row(j - 1).transpose() +
                             beta_mat * long_covs[i].row(j).transpose() +
                             gamma_mat * fixed_covs[i]);

        const VectorXd& dat = imputed_data[i].row(j).transpose();
        probas(h) += multi_normal_prec_lpdf(dat, currmean, sigma_inv);
      }
    }

    // if ((iter % 1000) == 0 && (i % 100) == 0 ) {
    //     std::cout << "probas before: " << probas.transpose() << std::endl;
    //     probas = probas.array().max(-10000);
    //     std::cout << "probas after: " << probas.transpose() << std::endl;
    //     probas = stan::math::softmax(probas);
    // } else {
    //     probas = probas.array().max(-10000);
    //     probas = stan::math::softmax(probas);
    // }

    probas = probas.array().max(-10000);
    probas = stan::math::softmax(probas);
    clus_allocs[i] =
        stan::math::categorical_rng(probas, Rng::Instance().get()) - 1;
  }
//   std::cout << "BaseSampler::sample_allocs() end" << std::endl;
}


// double BaseSampler::log_likelihood(MatrixXd y, MatrixXd phi, MatrixXd beta_mat,
//                                    MatrixXd gamma_mat, MatrixXd sigma_inv, 
//                                    MatrixXd long_cov, MatrixXd fixed_cov) {

//     double out = 0.0;
//     for (int j = 1; j < y.rows(); j++) {
//         VectorXd currmean = (phi * y.row(j - 1).transpose() +
//                              beta_mat * long_cov.row(j).transpose() +
//                              gamma_mat * fixed_cov);

//         const VectorXd& dat = y.row(j).transpose();
//         out += multi_normal_prec_lpdf(dat, currmean, sigma_inv);
//       }
//     return out;
// }

void BaseSampler::sample_dp() {
  // std::cout << "sample pg" << std::endl;
  if (H == 1) {
    dp_weights(0) = 1.0;
    return;
  }

  VectorXd data_by_clus = VectorXd::Zero(H);
  for (int i = 0; i < ndata; i++) {
    data_by_clus[clus_allocs[i]] += 1;
  }

  VectorXd stickbreaks(H - 1);
  for (int h = 0; h < H - 1; h++) {
    double a = 1.0 + data_by_clus(h);
    double b = dp_mass + data_by_clus.tail(H - (h + 1)).sum();
    stickbreaks(h) = stan::math::beta_rng(a, b, Rng::Instance().get());
  }

  dp_weights(0) = stickbreaks(0);
  dp_weights(H - 1) = 0;
  for (int h = 1; h < H - 1; h++)
    dp_weights(h) =
        stickbreaks(h) * (VectorXd::Ones(h) - stickbreaks.head(h)).prod();

  dp_weights(H - 1) = 1.0 - dp_weights.sum();
  dp_weights = (dp_weights.array() + 1e-6) /
               (dp_weights.sum() + 1e-6 * dp_weights.size());
}

// void BaseSampler::simulate_missing() {

//   MatrixXd sigma_inv_ = sigma_inv.get_prec();
//   using RowMat = Matrix<double, Dynamic, Dynamic, RowMajor>;

//   MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
//   MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

//   MatrixXd id_r = MatrixXd::Identity(rdim, rdim);
//   for (int j = 0; j < ndata; j++) {
//     if (is_missing[j].sum() == 0) continue;

//     MatrixXd phi = get_phi(j);

//     int l = orig_data[j].rows() - 1;
//     MatrixXd perm_mat =
//         get_permutation_matrix(is_missing[j].block(1, 0, l, rdim));

//     MatrixXd prec_mat = MatrixXd::Zero(l * rdim, l * rdim);
//     VectorXd mu = VectorXd::Zero(l * rdim);

//     VectorXd curr = phi * imputed_data[j].row(0).transpose() +
//                     beta_mat * long_covs[j].row(1).transpose() +
//                     gamma_mat * fixed_covs[j];
//     mu.head(rdim) = curr;
//     VectorXd prev = mu.head(rdim);
//     for (int i = 1; i < l; i++) {
//       curr = phi * prev + beta_mat * long_covs[j].row(i + 1).transpose() +
//              gamma_mat * fixed_covs[j];
//       mu.segment(i * rdim, rdim) = curr;
//       prev = curr;
//     }

//     MatrixXd diag_elem = (id_r + phi).transpose() * sigma_inv_ * (id_r + phi);
//     for (int i = 0; i < l - 1; i++) {
//       prec_mat.block(i * rdim, i * rdim, rdim, rdim) = diag_elem;
//       prec_mat.block(i * rdim, (i + 1) * rdim, rdim, rdim) =
//           phi.transpose() * sigma_inv_;
//       prec_mat.block((i + 1) * rdim, i * rdim, rdim, rdim) =
//           sigma_inv_.transpose() * phi;
//     }

//     prec_mat.block((l - 1) * rdim, (l - 1) * rdim, rdim, rdim) = sigma_inv_;

//     prec_mat = perm_mat * (prec_mat * perm_mat.transpose());
//     int k = is_missing[j].sum();

//     mu = perm_mat * mu;

//     VectorXd data_vect = vectorize(orig_data[j]);
//     data_vect = data_vect.tail(data_vect.size() - rdim);
//     data_vect = perm_mat * data_vect;

//     MatrixXd prec_missing = prec_mat.block(0, 0, k, k);
//     prec_missing = (prec_missing + prec_missing).array() * 0.5;
//     MatrixXd eps = MatrixXd::Identity(k, k).array() * 0.5;
//     prec_missing += eps;
//     // MatrixXd cov_missing = stan::math::inverse_spd(prec_missing);

//     VectorXd temp =
//         prec_mat.block(0, k, k, data_vect.size() - k) *
//         (data_vect.tail(data_vect.size() - k) - mu.tail(data_vect.size() - k));

//     // std::cout << "mu.head(k): " << mu.head(k).transpose() << std::endl;
//     VectorXd mean_missing = mu.head(k) - prec_missing.ldlt().solve(temp);

//     VectorXd sampled;
//     try {
//       sampled = stan::math::multi_normal_prec_rng(mean_missing, prec_missing,
//                                                   Rng::Instance().get());
//     } catch (...) {
//       sampled = mean_missing;
//     }

//     if (abs(sampled.maxCoeff()) > 1000) {
//       std::cout << "datum: " << j
//                 << ", mean_missing: " << mean_missing.transpose() << std::endl;
//       std::cout << "data - mu; "
//                 << (data_vect.tail(data_vect.size() - k) -
//                     mu.tail(data_vect.size() - k))
//                        .transpose()
//                 << std::endl;
//       std::cout << "temp: " << temp.transpose() << std::endl;
//       // std::cout << "cov_missing\n : " << cov_missing << std::endl;
//       VectorXd hack_mean = mu.head(k);
//       std::cout << "hack_mean: " << hack_mean.transpose() << std::endl;
//       // sampled = stan::math::multi_normal_prec_rng(hack_mean, prec_missing,
//       //                                             Rng::Instance().get());
//       sampled = hack_mean;
//     }

//     data_vect.head(k) = sampled;
//     data_vect = perm_mat.transpose() * data_vect;
//     // std::cout << "datum: "<< j << ", mean_missing: "
//     //           << mean_missing.transpose() << std::endl;
//     // std::cout << "data - mu; "
//     //           << (data_vect.tail(data_vect.size() - k) - \
//         //                  mu.tail(data_vect.size() - k)).transpose()
//     //           << std::endl;
//     // std::cout << "temp: " << temp.transpose() << std::endl;
//     // std::cout << "cov_missing\n : " << cov_missing << std::endl;
//     imputed_data[j].block(1, 0, l, rdim) =
//         Map<RowMat>(data_vect.data(), l, rdim);
//   }
//   // std::cout << "simulate_missing DONE" << std::endl;
// }

// std::vector<MatrixXd> BaseSampler::predict_phi(VectorXd fixed_covs,
//                                            std::deque<State> chains) {
  
//   int niter = chains.size();
//   std::vector<MatrixXd> out(niter);

//   for (int r = 0; r < niter; r++) {
//     out[r] = sample_predictive_phi(fixed_covs, chains[r], false);
//   }
//   return out;
// }

// std::vector<MatrixXd> BaseSampler::predict_one(VectorXd fixed_covs,
//                                            MatrixXd long_covs, VectorXd start,
//                                            std::deque<State> chains) {
//   std::cout << "BaseSampler::predict_one" << std::endl;
//   int niter = chains.size();
//   std::vector<MatrixXd> out(niter);
//   int ts = long_covs.rows();

//   for (int r = 0; r < niter; r++) {

//     restore_from_proto(chains[r]);
//     std::cout << "Restore from proto ok" << std::endl;
//     MatrixXd curr(ts, rdim);
//     curr.row(0) = start.transpose();

//     MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
//     MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
//     MatrixXd phi = sample_predictive_phi(fixed_covs, chains[r], true);
//     std::cout << "sample_predictive_phi ok" << std::endl;

//     for (int j = 1; j < ts; j++) {
//       VectorXd mean =
//           (phi * curr.row(j - 1).transpose() +
//            beta_mat * long_covs.row(j).transpose() + gamma_mat * fixed_covs)
//               .transpose();

//       curr.row(j) = mean;
//       // stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
//     }
//     out[r] = curr;
//   }
//   std::cout << "BaseSampler::predict_one DONE" << std::endl;
//   return out;
// }

// std::vector<MatrixXd> BaseSampler::predict_one_step(VectorXd fixed_covs,
//                                                 MatrixXd long_covs,
//                                                 MatrixXd vals,
//                                                 std::deque<State> chains) {
//   int niter = chains.size();
//   std::vector<MatrixXd> out(niter);
//   int ts = long_covs.rows();

//   for (int r = 0; r < niter; r++) {
//     restore_from_proto(chains[r]);

//     MatrixXd curr(ts, rdim);
//     curr.row(0) = vals.row(0);


//     MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
//     MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
//     MatrixXd phi = sample_predictive_phi(fixed_covs, chains[r], true);

//     for (int j = 1; j < ts; j++) {
//       VectorXd mean =
//           (phi * vals.row(j - 1).transpose() +
//            beta_mat * long_covs.row(j).transpose() + gamma_mat * fixed_covs)
//               .transpose();

//       curr.row(j) =
//           stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
//     }
//     out[r] = curr;
//   }
//   return out;
// }

// std::vector<MatrixXd> BaseSampler::predict_in_sample(int idx, int nsteps,
//                                                  VectorXd fixed_covs,
//                                                  MatrixXd long_covs,
//                                                  MatrixXd vals,
//                                                  std::deque<State> chains) {
//   int niter = chains.size();
//   std::vector<MatrixXd> out(niter);
//   int ts = long_covs.rows();
//   for (int r = 0; r < niter; r++) {
//     restore_from_proto(chains[r]);

//     MatrixXd curr(nsteps, rdim);
//     int clus = clus_allocs[idx];
//     VectorXd currval = vals.row(vals.rows() - 1).transpose();

//     MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
//     MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
//     MatrixXd phi = get_phi(idx);

//     for (int j = 0; j < ts; j++) {
//       VectorXd mean = (phi * currval + beta_mat * long_covs.row(j).transpose() +
//                        gamma_mat * fixed_covs)
//                           .transpose();

//       currval =
//           stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
//       curr.row(j) = currval.transpose();
//     }
//     out[r] = curr;
//   }
//   return out;
// }


MatrixXd BaseSampler::get_permutation_matrix(MatrixXi missing) {
  VectorXi missing_vec = vectorize(missing);
  MatrixXd out = MatrixXd::Zero(missing_vec.size(), missing_vec.size());
  std::vector<int> to_move;
  std::vector<int> leftovers;
  for (int i = 0; i < missing_vec.size(); i++) {
    if (missing_vec[i] > 0)
      to_move.push_back(i);
    else
      leftovers.push_back(i);
  }

  for (int i = 0; i < to_move.size(); i++) out(i, to_move[i]) = 1.0;

  for (int i = 0; i < leftovers.size(); i++)
    out(i + to_move.size(), leftovers[i]) = 1.0;

  return out;
}


void BaseSampler::get_state_as_proto(State* out) {
  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  to_proto(beta_mat, out->mutable_beta());

  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
  to_proto(gamma_mat, out->mutable_gamma());

  to_proto(sigma, out->mutable_sigma());

  *(out->mutable_clus_allocs()) = {clus_allocs.data(),
                                   clus_allocs.data() + ndata};
}

void BaseSampler::restore_from_proto(State state) {
  rdim = state.beta().rows();
  ldim = state.beta().cols();
  MatrixXd beta_mat = to_eigen(state.beta());
  beta_vec = Map<VectorXd>(beta_mat.data(), beta_mat.size());

  fdim = state.gamma().cols();
  MatrixXd gamma_mat = to_eigen(state.gamma());
  gamma_vec = Map<VectorXd>(gamma_mat.data(), gamma_mat.size());

  sigma = to_eigen(state.sigma());

  clus_allocs.resize(state.clus_allocs().size());
  for (int i = 0; i < clus_allocs.size(); i++)
    clus_allocs[i] = state.clus_allocs()[i];
}

// void BaseSampler::set_clus_alloc(std::vector<int> clus_alloc) {
//   std::cout << "set clus alloc" << std::endl;
//   this->clus_allocs = clus_alloc;
//   for (auto c : clus_alloc) std::cout << c << ", ";

//   std::cout << std::endl;
// }
