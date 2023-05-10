#include "sampler.hpp"

Sampler::Sampler(int H, std::vector<MatrixXd> data,
                 std::vector<MatrixXi> is_missing,
                 std::vector<MatrixXd> long_covs,
                 std::vector<VectorXd> fixed_covs, std::string weightsmodel)
    : BaseSampler(H, data, is_missing, long_covs, fixed_covs),
      weightsmodel(weightsmodel) {

  phis_vec.resize(H);
  pg_rng = new PolyaGammaHybridDouble(seed);
}

Sampler::Sampler(int H, std::vector<MatrixXd> data,
                 std::vector<MatrixXd> long_covs,
                 std::vector<VectorXd> fixed_covs, std::string weightsmodel)
    : BaseSampler(H, data, long_covs, fixed_covs), weightsmodel(weightsmodel) {
  phis_vec.resize(H);
  pg_rng = new PolyaGammaHybridDouble(seed);
}

void Sampler::set_base_measure(Eigen::MatrixXd phi00_, Eigen::MatrixXd v00_,
                               double lambda_, double tau0_) {
  if (phi00_.size() > 0) {
    bool dimension_check = phi00_.rows() == rdim;
    if (!dimension_check) {
      std::string err_msg =
          std::string("phi00_ must have dimension ") + std::to_string(rdim);
      throw std::invalid_argument(err_msg);
    }
  } else {
    phi00_ = Eigen::MatrixXd::Zero(rdim, rdim);
  }
  if (v00_.size() > 0) {
    if (v00_.rows() != rdim * rdim) {
      std::string err_msg = std::string("v00_ must have dimension ") +
                            std::to_string(rdim * rdim);
      throw std::invalid_argument(err_msg);
    }
  } else {
    v00_ = Eigen::MatrixXd::Identity(rdim * rdim, rdim * rdim);
  }

  phi_00 = Map<VectorXd>(phi00_.data(), phi00_.size());
  v_00 = v00_;
  lambda = lambda_;
  tau_0 = tau0_;
}

void Sampler::set_prior_lsb(Eigen::VectorXd mu_alpha_, double var) {
  if (mu_alpha_.size() > 0) {
    if (mu_alpha_.size() != fdim) {
      std::string err_msg =
          std::string("mu_alpha must have dimension ") + std::to_string(fdim);
      throw std::invalid_argument(err_msg);
    }
    mu_alpha = mu_alpha_;
  } else {
    mu_alpha = VectorXd::Zero(fdim);
  }
  sigma_alpha = MatrixXd::Identity(fdim, fdim).array() * var;
  sigma_alpha_inv = MatrixXd::Identity(fdim, fdim).array() / var;
}

void Sampler::initialize() {
  BaseSampler::initialize();
 
  v_0 = v_00 / (tau_0 - rdim * rdim - 1);
  v_0_inv = stan::math::inverse_spd(v_0);
  std::cout << "v0: \n" << v_0 << std::endl;
  phi_0 = phi_00;

  for (int h = 0; h < H; h++) {
    phis_vec[h] = phi_0;
  }

  alphas.resize(H);
  for (int h = 0; h < H; h++) {
    alphas[h] = stan::math::multi_normal_rng(mu_alpha, sigma_alpha,
                                             Rng::Instance().get());
  }
  dp_weights = VectorXd::Ones(H) / (1.0 * H);
  if (missing_data) simulate_missing();
}

void Sampler::step() {
  iter++;
  // sample_beta();
  sample_sigma();
  // sample_gamma();
  sample_phis();
  if (weightsmodel == "LSB") {
    sample_pg();
  } else if (weightsmodel == "DP") {
    sample_dp();
  }
  sample_allocs();
  sample_hypers();
  relabel();

  if (missing_data) simulate_missing();
}

void Sampler::sample_phis() {
  // std::cout << "sample phis" << std::endl;

  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

  // #pragma omp parallel for
  for (int h = 0; h < H; h++) {
    // std::cout << "h: " << h << std::endl;
    std::vector<MatrixXd> data_by_clus;
    std::vector<MatrixXd> long_covs_by_clus;
    std::vector<VectorXd> fixed_covs_by_clus;

    // std::cout << "data_index: ";
    for (int i = 0; i < ndata; i++) {
      if (clus_allocs[i] == h) {
        // std::cout << i << ", ";
        data_by_clus.push_back(imputed_data[i]);
        long_covs_by_clus.push_back(long_covs[i]);
        fixed_covs_by_clus.push_back(fixed_covs[i]);
      }
    }
    // std::cout << std::endl;

    if (data_by_clus.size() == 0) {
      phis_vec[h] =
          stan::math::multi_normal_rng(phi_0, v_0, Rng::Instance().get());
      MatrixXd phi_mat = Map<MatrixXd>(phis_vec[h].data(), rdim, rdim);
      // std::cout << "sampled from prior " << std::endl
      //           << phi_mat << std::endl
      //           << std::endl;
    } else {
      int nsteps = 0;
      for (int i = 0; i < data_by_clus.size(); i++)
        nsteps += data_by_clus[i].rows();

      MatrixXd curr_data(nsteps - data_by_clus.size(), rdim);
      MatrixXd curr_preds(nsteps - data_by_clus.size(), rdim);
      MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
      MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
      int row = 0;
      for (int i = 0; i < data_by_clus.size(); i++) {
        int l = data_by_clus[i].rows();
        for (int j = 1; j < l; j++) {
          curr_data.row(row) =
              (data_by_clus[i].row(j).transpose() -
               beta_mat * long_covs_by_clus[i].row(j).transpose() -
               gamma_mat * fixed_covs_by_clus[i])
                  .transpose();
          curr_preds.row(row) = data_by_clus[i].row(j - 1);
          row += 1;
        }
      }

      //   std::cout << "curr_data \n" << curr_data << std::endl;
      //   std::cout << "curr_preds \n" << curr_preds << std::endl;

      // if (abs(curr_preds.maxCoeff()) > 50)
      //     std::cout << "curr_preds: \n" << curr_preds << std::endl;

      MatrixXd xtransx = curr_preds.transpose() * curr_preds;
      // std::cout << "xtransx: \n " << xtransx << std::endl;
      // assert(abs(xtransx.maxCoeff()) < 1e4);

      MatrixXd phi_hat_ =
          xtransx.ldlt().solve(curr_preds.transpose() * curr_data);
      // std::cout << "phi_hat_ \n" << phi_hat_ << std::endl;

      VectorXd phi_hat = Map<VectorXd>(phi_hat_.data(), rdim * rdim);
      // std::cout << "sigma_inv \n: " << sigma_inv << std::endl;;
      MatrixXd helper = kroneckerProduct(sigma_inv.get_prec(), xtransx);

      MatrixXd prec_phi_post = (helper + v_0_inv);
      // prec_phi_post = (prec_phi_post + prec_phi_post.transpose()).array() *
      // 0.5; int k = prec_phi_post.rows(); MatrixXd eps = MatrixXd::Identity(k,
      // k).array() * 0.1; prec_phi_post += eps;

      VectorXd mu_phi_post =
          prec_phi_post.ldlt().solve(helper * phi_hat + v_0_inv * phi_0);

      phis_vec[h] = stan::math::multi_normal_prec_rng(
          mu_phi_post, prec_phi_post, Rng::Instance().get());

      // std::cout << "Sampled phi\n"
      //           << Map<MatrixXd>(phis_vec[h].data(), rdim, rdim) << std::endl
      //           << std::endl;
    }
  }
  // std::cout << "sample phis DONE" << std::endl;
}

// void Sampler::sample_allocs() {
//   MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
//   MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

// #pragma omp parallel for
//   for (int i = 0; i < ndata; i++) {
//     // if (adapt && is_missing[i].size() > 0) {
//     //   continue;
//     // }

//     VectorXd probas = get_weights(fixed_covs[i], true, true);

//     // std::cout << "datum: " << i << ", weights: " << probas.transpose()
//     //           << std::endl;
//     // VectorXd probas = VectorXd::Ones(H).array() / H;

//     for (int h = 0; h < H; h++) {
//       MatrixXd phi = Map<MatrixXd>(phis_vec[h].data(), rdim, rdim);
//       for (int j = 1; j < imputed_data[i].rows(); j++) {
//         VectorXd currmean = (phi * imputed_data[i].row(j - 1).transpose() +
//                              beta_mat * long_covs[i].row(j).transpose() +
//                              gamma_mat * fixed_covs[i]);

//         const VectorXd& dat = imputed_data[i].row(j).transpose();
//         probas(h) += multi_normal_prec_lpdf(dat, currmean, sigma_inv);
//       }
//     }

//     probas = stan::math::softmax(probas);

//     clus_allocs[i] =
//         stan::math::categorical_rng(probas, Rng::Instance().get()) - 1;
//   }
//   // std::cout << "clus_allocs: ";
//   // for (auto c : clus_allocs) std::cout << c << ", ";
//   // std::cout << std::endl;
// }

void Sampler::sample_pg() {
  // std::cout << "sample pg" << std::endl;

  for (int h = 0; h < H - 1; h++) {
    std::vector<bool> tosample(ndata);
    for (int i = 0; i < ndata; i++) {
      tosample[i] = (clus_allocs[i] > h - 1);
    }
    int n = std::accumulate(tosample.begin(), tosample.end(), 0.0);
    MatrixXd omegas = MatrixXd::Zero(n, n);
    MatrixXd curr_covs = MatrixXd::Zero(n, fdim);
    VectorXd kappa = VectorXd::Zero(n);
    int pos = 0;
    for (int i = 0; i < ndata; i++) {
      if (tosample[i]) {
        omegas(pos, pos) =
            pg_rng->draw(1, fixed_covs[i].transpose() * alphas[h]);
        curr_covs.row(pos) = fixed_covs[i].transpose();
        if (clus_allocs[i] == h)
          kappa[pos] = 0.5;
        else
          kappa[pos] = -0.5;
        pos += 1;
      }
    }

    MatrixXd sigma_alpha_post =
        curr_covs.transpose() * omegas * curr_covs + sigma_alpha_inv;

    VectorXd mu_alpha_post = sigma_alpha_post.ldlt().solve(
        curr_covs.transpose() * kappa + sigma_alpha_inv * mu_alpha);

    // std::cout << "mu_alpha_post: " << mu_alpha_post.transpose() << std::endl;

    alphas[h] = stan::math::multi_normal_prec_rng(
        mu_alpha_post, sigma_alpha_post, Rng::Instance().get());
  }

  // std::cout << "sample_pg done" << std::endl;
}


void Sampler::relabel() {
  // Sort the clusters from the most popoulos to the least one, to help
  // the Polya-Gamma trick

  std::vector<int> cardinalities(H, 0);
  std::vector<int> pos(H);
  std::iota(pos.begin(), pos.end(), 0);

  for (const int c : clus_allocs) cardinalities[c] += 1;

  std::sort(pos.begin(), pos.end(),
            [&](int i, int j) { return cardinalities[i] > cardinalities[j]; });

  std::map<int, int> old2new;
  for (int i = 0; i < H; i++) old2new.insert(std::pair<int, int>(pos[i], i));

  std::vector<VectorXd> new_phis(H);
  std::vector<VectorXd> new_alphas(H);
  std::vector<int> new_allocs(ndata);

  for (int h = 0; h < H; h++) {
    new_phis[h] = phis_vec[pos[h]];
    new_alphas[h] = alphas[pos[h]];
  }

  for (int i = 0; i < ndata; i++) {
    new_allocs[i] = old2new[clus_allocs[i]];
  }

  phis_vec = new_phis;
  alphas = new_alphas;
  clus_allocs = new_allocs;
}

void Sampler::sample_hypers() {
  std::vector<VectorXd> used_phis;
  for (int h = 0; h < H; h++) {
    if (std::count(clus_allocs.begin(), clus_allocs.end(), h) > 0) {
      used_phis.push_back(phis_vec[h]);
    }
  }
  double n_ = 1.0 * used_phis.size();

  VectorXd phi_bar = VectorXd::Zero(phis_vec[0].size());
  MatrixXd sum_squares = MatrixXd::Zero(phis_vec[0].size(), phis_vec[0].size());
  for (const auto& phi : used_phis) {
    phi_bar += phi;
    sum_squares += phi * phi.transpose();
  }
  phi_bar = phi_bar.array() / n_;
  sum_squares -= phi_bar * phi_bar.transpose();
  VectorXd post_mean = (lambda * phi_00 + n_ * phi_bar) / (lambda + n_);
  MatrixXd post_var = v_00 + sum_squares +
                      (phi_bar - phi_00) * (phi_bar - phi_00).transpose() *
                          lambda * n_ / (lambda + n_);
  v_0 = stan::math::inv_wishart_rng(nu + n_, post_var, Rng::Instance().get());
  // std::cout << "v0_mean: \n " << post_var.array() / (nu + n_ - post_var.rows() - 1) << std::endl;
  // std::cout << "v0 \n" << v_0 << std::endl;
  phi_0 = stan::math::multi_normal_rng(post_mean, v_0 / (lambda + n_),
                                       Rng::Instance().get());
  // std::cout << "n_" << n_ << ", phi_0: " << phi_0.transpose() << std::endl;
}

void Sampler::simulate_missing() {
  // std::cout << "simulate_missing" << std::endl;

  MatrixXd sigma_inv_ = sigma_inv.get_prec();
  using RowMat = Matrix<double, Dynamic, Dynamic, RowMajor>;

  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

  MatrixXd id_r = MatrixXd::Identity(rdim, rdim);
  for (int j = 0; j < ndata; j++) {
    if (is_missing[j].sum() == 0) continue;

    // MatrixXd phi = Map<MatrixXd>(phis_vec[clus_allocs[j]].data(), rdim,
    // rdim);
    MatrixXd phi = Map<MatrixXd>(phi_00.data(), rdim, rdim);

    // std::cout << "phi \n" << phi << std::endl;

    int l = orig_data[j].rows() - 1;
    MatrixXd perm_mat =
        get_permutation_matrix(is_missing[j].block(1, 0, l, rdim));

    MatrixXd prec_mat = MatrixXd::Zero(l * rdim, l * rdim);
    VectorXd mu = VectorXd::Zero(l * rdim);

    VectorXd curr = phi * imputed_data[j].row(0).transpose() +
                    beta_mat * long_covs[j].row(1).transpose() +
                    gamma_mat * fixed_covs[j];
    mu.head(rdim) = curr;
    VectorXd prev = mu.head(rdim);
    for (int i = 1; i < l; i++) {
      curr = phi * prev + beta_mat * long_covs[j].row(i + 1).transpose() +
             gamma_mat * fixed_covs[j];
      mu.segment(i * rdim, rdim) = curr;
      prev = curr;
    }

    MatrixXd diag_elem = (id_r + phi).transpose() * sigma_inv_ * (id_r + phi);
    for (int i = 0; i < l - 1; i++) {
      prec_mat.block(i * rdim, i * rdim, rdim, rdim) = diag_elem;
      prec_mat.block(i * rdim, (i + 1) * rdim, rdim, rdim) =
          phi.transpose() * sigma_inv_;
      prec_mat.block((i + 1) * rdim, i * rdim, rdim, rdim) =
          sigma_inv_.transpose() * phi;
    }

    prec_mat.block((l - 1) * rdim, (l - 1) * rdim, rdim, rdim) = sigma_inv_;

    prec_mat = perm_mat * (prec_mat * perm_mat.transpose());
    int k = is_missing[j].sum();

    mu = perm_mat * mu;

    VectorXd data_vect = vectorize(orig_data[j]);
    data_vect = data_vect.tail(data_vect.size() - rdim);
    data_vect = perm_mat * data_vect;

    MatrixXd prec_missing = prec_mat.block(0, 0, k, k);
    prec_missing = (prec_missing + prec_missing).array() * 0.5;
    MatrixXd eps = MatrixXd::Identity(k, k).array() * 0.5;
    prec_missing += eps;
    // MatrixXd cov_missing = stan::math::inverse_spd(prec_missing);

    VectorXd temp =
        prec_mat.block(0, k, k, data_vect.size() - k) *
        (data_vect.tail(data_vect.size() - k) - mu.tail(data_vect.size() - k));

    // std::cout << "mu.head(k): " << mu.head(k).transpose() << std::endl;
    VectorXd mean_missing = mu.head(k) - prec_missing.ldlt().solve(temp);

    VectorXd sampled;
    try {
      sampled = stan::math::multi_normal_prec_rng(mean_missing, prec_missing,
                                                  Rng::Instance().get());
    } catch (...) {
      sampled = mean_missing;
    }

    if (abs(sampled.maxCoeff()) > 1000) {
      std::cout << "datum: " << j
                << ", mean_missing: " << mean_missing.transpose() << std::endl;
      std::cout << "data - mu; "
                << (data_vect.tail(data_vect.size() - k) -
                    mu.tail(data_vect.size() - k))
                       .transpose()
                << std::endl;
      std::cout << "temp: " << temp.transpose() << std::endl;
      // std::cout << "cov_missing\n : " << cov_missing << std::endl;
      VectorXd hack_mean = mu.head(k);
      std::cout << "hack_mean: " << hack_mean.transpose() << std::endl;
      // sampled = stan::math::multi_normal_prec_rng(hack_mean, prec_missing,
      //                                             Rng::Instance().get());
      sampled = hack_mean;
    }

    data_vect.head(k) = sampled;
    data_vect = perm_mat.transpose() * data_vect;
    // std::cout << "datum: "<< j << ", mean_missing: "
    //           << mean_missing.transpose() << std::endl;
    // std::cout << "data - mu; "
    //           << (data_vect.tail(data_vect.size() - k) - \
        //                  mu.tail(data_vect.size() - k)).transpose()
    //           << std::endl;
    // std::cout << "temp: " << temp.transpose() << std::endl;
    // std::cout << "cov_missing\n : " << cov_missing << std::endl;
    imputed_data[j].block(1, 0, l, rdim) =
        Map<RowMat>(data_vect.data(), l, rdim);
  }
  // std::cout << "simulate_missing DONE" << std::endl;
}

VectorXd Sampler::lsb_weights(VectorXd covs, bool log_, bool train_) {
  VectorXd nus = VectorXd::Zero(H);
  VectorXd lognus = VectorXd::Zero(H);
  VectorXd log_1mnus = VectorXd::Zero(H);

  VectorXd out = VectorXd::Zero(H);
  VectorXd logout = VectorXd::Zero(H);

  if (adapt) {
    out = VectorXd::Ones(H).array() / H;
  } else {
    for (int h = 0; h < H - 1; h++) {
      double x = covs.transpose() * alphas[h];
      nus(h) = 1.0 / (1.0 + std::exp(-x));
      lognus(h) = -std::log1p(std::exp(-x));
      log_1mnus(h) = -x + lognus(h);
    }
    nus(H-1) = 1.0;
    lognus(H-1) = 0.0;

    out(0) = nus(0);
    logout(0) = lognus(0);
    for (int h = 1; h < H; h++) {
      out(h) = nus(h) * (VectorXd::Ones(h) - nus.head(h)).prod();
      logout(h) = lognus(h) + log_1mnus.head(h).sum();
    }

    if (train_) {
      out = out.array() + 1e-12;
      logout = logout.array().max(std::log(1e-12));
    }

  }

  if (log_) {
    return logout;
  } else {
    return out;
  }
}

void Sampler::get_state_as_proto(State* out) {
  BaseSampler::get_state_as_proto(out);
  
  for (int h = 0; h < H; h++) {
    MatrixXd phi = Map<MatrixXd>(phis_vec[h].data(), rdim, rdim);
    EigenMatrix* phi_proto = out->add_phis();
    to_proto(phi, phi_proto);
  }
  out->set_model(weightsmodel);
  if (weightsmodel == "LSB") {
    for (int h = 0; h < H; h++) {
      EigenVector* alpha_proto = out->add_alphas();
      to_proto(alphas[h], alpha_proto);
    }
  } else if (weightsmodel == "DP") {
    to_proto(dp_weights, out->mutable_dp_weights());
  }
}

void Sampler::restore_from_proto(State state) {
  BaseSampler::restore_from_proto(state);

  H = state.phis().size();
  phis_vec.resize(H);
  alphas.resize(H);
  for (int h = 0; h < H; h++) {
    MatrixXd phi_mat = to_eigen(state.phis()[h]);
    phis_vec[h] = Map<VectorXd>(phi_mat.data(), phi_mat.size());
  }

  weightsmodel = state.model();

  if (weightsmodel == "LSB") {
    for (int h = 0; h < H; h++) alphas[h] = to_eigen(state.alphas()[h]);
  } else if (weightsmodel == "DP") {
    dp_weights = to_eigen(state.dp_weights());
  }
}

std::vector<MatrixXd> Sampler::predict_phi(VectorXd fixed_covs,
                                           std::deque<State> chains) {
  int niter = chains.size();
  std::vector<MatrixXd> out(niter);

  for (int r = 0; r < niter; r++) {
    restore_from_proto(chains[r]);
    VectorXd clus_probas = get_weights(fixed_covs, false);
    int clus =
        stan::math::categorical_rng(clus_probas, Rng::Instance().get()) - 1;
    MatrixXd phi = Map<MatrixXd>(phis_vec[clus].data(), rdim, rdim);
    out[r] = phi;
    std::cout << "phi\n" << phi << std::endl;
  }
  return out;
}

std::vector<MatrixXd> Sampler::predict_one(VectorXd fixed_covs,
                                           MatrixXd long_covs, VectorXd start,
                                           std::deque<State> chains) {
  int niter = chains.size();
  std::vector<MatrixXd> out(niter);
  int ts = long_covs.rows();

  for (int r = 0; r < niter; r++) {
    restore_from_proto(chains[r]);

    MatrixXd curr(ts, rdim);
    curr.row(0) = start.transpose();

    VectorXd clus_probas = get_weights(fixed_covs, false);

    int clus =
        stan::math::categorical_rng(clus_probas, Rng::Instance().get()) - 1;

    MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
    MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
    MatrixXd phi = Map<MatrixXd>(phis_vec[clus].data(), rdim, rdim);

    if (r == niter - 1) {
      std::cout << "Final Gamma \n" << gamma_mat << std::endl;
    }

    for (int j = 1; j < ts; j++) {
      VectorXd mean =
          (phi * curr.row(j - 1).transpose() +
           beta_mat * long_covs.row(j).transpose() + gamma_mat * fixed_covs)
              .transpose();

      curr.row(j) = mean;
      // stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
    }
    out[r] = curr;
  }
  return out;
}

std::vector<MatrixXd> Sampler::predict_one_step(VectorXd fixed_covs,
                                                MatrixXd long_covs,
                                                MatrixXd vals,
                                                std::deque<State> chains) {
  int niter = chains.size();
  std::vector<MatrixXd> out(niter);
  int ts = long_covs.rows();

  for (int r = 0; r < niter; r++) {
    restore_from_proto(chains[r]);

    MatrixXd curr(ts, rdim);
    curr.row(0) = vals.row(0);

    VectorXd clus_probas = get_weights(fixed_covs, false);

    int clus =
        stan::math::categorical_rng(clus_probas, Rng::Instance().get()) - 1;

    MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
    MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
    MatrixXd phi = Map<MatrixXd>(phis_vec[clus].data(), rdim, rdim);

    if (r == niter - 1) {
      std::cout << "Final Gamma \n" << gamma_mat << std::endl;
    }

    for (int j = 1; j < ts; j++) {
      VectorXd mean =
          (phi * vals.row(j - 1).transpose() +
           beta_mat * long_covs.row(j).transpose() + gamma_mat * fixed_covs)
              .transpose();

      curr.row(j) =
          stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
    }
    out[r] = curr;
  }
  return out;
}

std::vector<MatrixXd> Sampler::predict_in_sample(int idx, int nsteps,
                                                 VectorXd fixed_covs,
                                                 MatrixXd long_covs,
                                                 MatrixXd vals,
                                                 std::deque<State> chains) {
  int niter = chains.size();
  std::vector<MatrixXd> out(niter);
  int ts = long_covs.rows();
  for (int r = 0; r < niter; r++) {
    restore_from_proto(chains[r]);

    MatrixXd curr(nsteps, rdim);
    int clus = clus_allocs[idx];
    VectorXd currval = vals.row(vals.rows() - 1).transpose();

    MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
    MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
    MatrixXd phi = Map<MatrixXd>(phis_vec[clus].data(), rdim, rdim);

    if (r == niter - 1) {
      std::cout << "Final Gamma \n" << gamma_mat << std::endl;
    }

    for (int j = 0; j < ts; j++) {
      VectorXd mean = (phi * currval + beta_mat * long_covs.row(j).transpose() +
                       gamma_mat * fixed_covs)
                          .transpose();

      currval =
          stan::math::multi_normal_rng(mean, sigma, Rng::Instance().get());
      curr.row(j) = currval.transpose();
    }
    out[r] = curr;
  }
  return out;
}

VectorXd Sampler::get_weights(VectorXd fixed_covs, bool log_, bool train_) {
  VectorXd clus_probas;
  if (weightsmodel == "LSB") {
    clus_probas = lsb_weights(fixed_covs, log_, train_);
  } else if (weightsmodel == "DP") {
    if (log_)
      clus_probas = log(dp_weights.array());
    else
      clus_probas = dp_weights;
  }
  return clus_probas;
}

void Sampler::set_clus_alloc(std::vector<int> clus_alloc) {
  std::cout << "set clus alloc" << std::endl;
  this->clus_allocs = clus_alloc;
  for (auto c : clus_alloc) std::cout << c << ", ";

  std::cout << std::endl;
}
