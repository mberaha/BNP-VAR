#include "linear_ddp_sampler.hpp"

void LinearDDPSampler::set_base_measure(VectorXd beta0, MatrixXd cov) {
    if (beta0.size() != rdim * rdim * fdim) {
        std::cout << "Found inconsistent dimension of beta0 - initializing with zero" << std::endl;
        int dim = rdim * rdim * fdim;
        cluster_reg_mean = VectorXd::Zero(dim);
        cluster_reg_cov = MatrixXd::Identity(dim, dim);
    } else {
        cluster_reg_mean = beta0;
        cluster_reg_cov = cov;
    }
}

void LinearDDPSampler::initialize() {
  BaseSampler::initialize();

  std::cout << "Initializing Linear DDP sampler" << std::endl;
  cluster_regressors.resize(H);
  for (int h = 0; h < H; h++) {
    cluster_regressors[h] = cluster_reg_mean;
  }
  std::cout << "h" << std::endl;
  
  int nh = cluster_reg_mean.size();
  mh_prop_cov = Eigen::MatrixXd::Identity(nh, nh) * 0.1;
  mh_rolling_mean = cluster_reg_mean;

  dp_weights = VectorXd::Ones(H) / (1.0 * H);
  if (missing_data) simulate_missing();

  std::cout << "Done initializing Linear DDP sampler" << std::endl;
}

void LinearDDPSampler::step() {
  iter++;
//   sample_beta();
//   sample_sigma();
//   sample_gamma();
  sample_cluster_regressors();
  sample_dp();
  sample_allocs();
  if (missing_data) simulate_missing();
}

void LinearDDPSampler::sample_cluster_regressors() {
  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

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

    if (data_by_clus.size() == 0) {
      cluster_regressors[h] =
          stan::math::multi_normal_rng(cluster_reg_mean, cluster_reg_cov, Rng::Instance().get());
    } else {
      Eigen::VectorXd prop = stan::math::multi_normal_rng(
          cluster_regressors[h], mh_prop_cov, Rng::Instance().get());
      
      double prior_lpdf_prop = stan::math::multi_normal_lpdf(prop, cluster_reg_mean, cluster_reg_cov);
      double prior_lpdf_curr = stan::math::multi_normal_lpdf(
          cluster_regressors[h], cluster_reg_mean, cluster_reg_cov);
      double like_lpdf_prop = 0;
      double like_lpdf_curr = 0;

      for (int i=0; i < data_by_clus.size(); i++) {
          MatrixXd phi_curr = get_phi(h, fixed_covs_by_clus[i]);
          MatrixXd phi_prop = get_phi(prop, fixed_covs_by_clus[i]);

          for (int j = 1; j < data_by_clus[i].rows(); j++) {
           const VectorXd& dat = data_by_clus[i].row(j).transpose();
           VectorXd fixmean = gamma_mat * fixed_covs_by_clus[i];
           VectorXd longmean = beta_mat * long_covs_by_clus[i].row(j).transpose();
           like_lpdf_prop += multi_normal_prec_lpdf(
               dat, phi_prop * data_by_clus[i].row(j- 1).transpose() +
                longmean + fixmean, sigma_inv);
            like_lpdf_curr += multi_normal_prec_lpdf(
               dat, phi_curr * data_by_clus[i].row(j- 1).transpose() +
                longmean + fixmean, sigma_inv);
        }
      }

      double arate = like_lpdf_prop + prior_lpdf_prop -
          like_lpdf_curr - prior_lpdf_curr;

      if (stan::math::uniform_rng(0, 1, Rng::Instance().get()) < arate) {
        // std::cout << "Accepted" << std::endl;
        cluster_regressors[h] = prop;
      }
    
      Eigen::VectorXd tmp = cluster_regressors[h] - mh_rolling_mean;
      mh_rolling_mean = mh_rolling_mean +  tmp * adapt_step(iter);
      mh_prop_cov = mh_prop_cov + (tmp * tmp.transpose() - mh_prop_cov) * adapt_step(iter);
    }
  }
}


void LinearDDPSampler::simulate_missing() {
  MatrixXd sigma_inv_ = sigma_inv.get_prec();
  using RowMat = Matrix<double, Dynamic, Dynamic, RowMajor>;

  MatrixXd beta_mat = Map<MatrixXd>(beta_vec.data(), rdim, ldim);
  MatrixXd gamma_mat = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);

  MatrixXd id_r = MatrixXd::Identity(rdim, rdim);
  for (int j = 0; j < ndata; j++) {
    if (is_missing[j].sum() == 0) continue;

    MatrixXd phi = get_phi(clus_allocs[j], fixed_covs[j]);
    int l = orig_data[j].rows() - 1;
    MatrixXd perm_mat =
        get_permutation_matrix(is_missing[j].block(1, 0, l, rdim));

    MatrixXd prec_mat = MatrixXd::Zero(l * rdim, l * rdim);
    VectorXd mu = VectorXd::Zero(l * rdim);

    VectorXd fixmean = gamma_mat * fixed_covs[j];
    VectorXd longmean = beta_mat * long_covs[j].row(1).transpose();
    VectorXd armean = phi * orig_data[j].row(0).transpose();

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

    VectorXd mean_missing = mu.head(k) - prec_missing.ldlt().solve(temp);

    VectorXd sampled;
    try {
      sampled = stan::math::multi_normal_prec_rng(mean_missing, prec_missing,
                                                  Rng::Instance().get());
    } catch (...) {
      sampled = mean_missing;
    }

    if (abs(sampled.maxCoeff()) > 1000) {
    //   std::cout << "datum: " << j
    //             << ", mean_missing: " << mean_missing.transpose() << std::endl;
    //   std::cout << "data - mu; "
    //             << (data_vect.tail(data_vect.size() - k) -
    //                 mu.tail(data_vect.size() - k))
    //                    .transpose()
    //             << std::endl;
    //   std::cout << "temp: " << temp.transpose() << std::endl;
      // std::cout << "cov_missing\n : " << cov_missing << std::endl;
      VectorXd hack_mean = mu.head(k);
    //   std::cout << "hack_mean: " << hack_mean.transpose() << std::endl;
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
}


void LinearDDPSampler::get_state_as_proto(State* out) {
  BaseSampler::get_state_as_proto(out);
  
  to_proto(dp_weights, out->mutable_dp_weights());
  for (int h = 0; h < H; h++) {
    EigenVector* beta_proto = out->add_lindpp_regressors();
    to_proto(cluster_regressors[h], beta_proto);
  }

}

void LinearDDPSampler::restore_from_proto(State state) {
    BaseSampler::restore_from_proto(state);
}