#include <Eigen/Dense>
#include <deque>
#include <stan/math/prim.hpp>
#include <vector>

#include "../rng.hpp"
#include "../sampler.hpp"

using namespace Eigen;

int main() {
  MatrixXd beta_true = MatrixXd::Identity(3, 3);
  MatrixXd gamma_true = MatrixXd::Zero(3, 2);
  // MatrixXd gamma_true(3, 2);
  // gamma_true << 0, 1, 0.5, 1, 0, 0;
  MatrixXd sigma_true = MatrixXd::Identity(3, 3).array() * 0.25;
  VectorXd alpha1(2);
  alpha1 << 10.0, 10.0;
  VectorXd alpha2(2);
  alpha2 << -10.0, -10.0;

  MatrixXd phi1 = MatrixXd::Identity(3, 3);
  MatrixXd phi2 = MatrixXd::Zero(3, 3);

  int ndata = 100;
  int ts = 10;
  std::vector<MatrixXd> orig_data(ndata);
  std::vector<MatrixXd> long_covs(ndata);
  std::vector<VectorXd> fixed_covs(ndata);
  std::vector<MatrixXi> is_missing(ndata);

  VectorXd mean_fixed = VectorXd::Zero(2);
  MatrixXd cov_fixed = MatrixXd::Identity(2, 2).array();

  VectorXd mean_long = VectorXd::Zero(3);
  MatrixXd cov_long = MatrixXd::Identity(3, 3);

  std::vector<int> true_clus(ndata);

  for (int i = 0; i < ndata; i++) {
    fixed_covs[i] = stan::math::multi_normal_rng(mean_fixed, cov_fixed,
                                                 Rng::Instance().get());

    VectorXd clus_probas(2);
    clus_probas(0) =
        1.0 / (1.0 + std::exp(-alpha1.transpose() * fixed_covs[i]));
    clus_probas(1) =
        1.0 / (1.0 + std::exp(-alpha2.transpose() * fixed_covs[i]));
    // std::cout << "clus_probas: " << clus_probas.transpose() << std::endl;
    if (clus_probas(0) > clus_probas(1))
      true_clus[i] = 0;
    else
      true_clus[i] = 1;

    long_covs[i].resize(ts, 3);
    long_covs[i].row(0) = stan::math::multi_normal_rng(mean_long, cov_long,
                                                       Rng::Instance().get());

    orig_data[i].resize(ts, 3);
    orig_data[i].row(0) = stan::math::multi_normal_rng(mean_long, cov_long,
                                                       Rng::Instance().get());

    for (int j = 1; j < ts; j++) {
      long_covs[i].row(j) = stan::math::multi_normal_rng(mean_long, cov_long,
                                                         Rng::Instance().get());
      VectorXd mean = beta_true * long_covs[i].row(j).transpose() +
                      gamma_true * fixed_covs[i];

      if (true_clus[i] == 0) {
        mean += phi1 * orig_data[i].row(j - 1).transpose();
      } else {
        mean += phi2 * orig_data[i].row(j - 1).transpose();
      }
      orig_data[i].row(j) =
          stan::math::multi_normal_rng(mean, sigma_true, Rng::Instance().get())
              .transpose();
    }

    MatrixXi miss = MatrixXi::Zero(ts, 3);
    miss(5, 1) = 1;
    miss(2, 2) = 1;
    is_missing[i] = miss;
  }

  Sampler sampler(5, orig_data, is_missing, long_covs, fixed_covs);

  sampler.set_base_measure();
  sampler.set_prior_sigma();
  sampler.set_prior_beta();
  sampler.set_prior_gamma();
  sampler.set_prior_lsb();
  sampler.set_prior_dp();

  std::cout << "initializing" << std::endl;

  sampler.initialize();

  std::cout << "done initializing" << std::endl;
  // sampler.set_cluster(true_clus);
  // sampler.set_beta(beta_true);
  // sampler.set_gamma(gamma_true);
  // sampler.set_sigma(sigma_true);

  std::deque<State> chains;

  for (int i = 0; i < 50000; i++) {
    sampler.step();
    if (i + 1 % 1000 == 0)
      std::cout << "Burn in: " << i + 1 << " / " << 50000 << std::endl;
  }

  for (int i = 0; i < 5000; i++) {
    sampler.step();
    State curr;
    sampler.get_state_as_proto(&curr);
    chains.push_back(curr);
  }

  // std::vector<MatrixXd> imputed_data = sampler.get_imputed_data();
  // for (int i=0; i < ndata; i++) {
  //   std::cout << "True: " << orig_data[i](2, 2) << ", " << orig_data[i](5, 1)
  //             << "    Imputed: " << imputed_data[i](2, 2) << ", "
  //             << imputed_data[i](5, 1) << std::endl;
  // }

  // std::cout << "Orig Data: " << std::endl;
  // for (const auto& datum : orig_data) {
  //   std::cout << "d: " << std::endl;
  //   std::cout << datum << std::endl;
  // }

  // std::cout << "Imputed Data: " << std::endl;
  // for (const auto& datum : sampler.get_imputed_data()) {
  //   std::cout << "d: " << std::endl;
  //   std::cout << datum << std::endl;
  // }

  std::cout << "true cluster: ";
  for (const auto& c : true_clus) std::cout << c << ", ";
  std::cout << std::endl;

  std::cout << "cluster: ";
  for (const auto& c : sampler.get_cluster()) std::cout << c << ", ";
  std::cout << std::endl;

  // std::cout << "phis: " << std::endl;
  // for (const auto &phi : sampler.get_phis())
  //     std::cout << phi.transpose() << std::endl << std::endl;

  // std::cout << "alphas: " << std::endl;
  // for (const auto &phi : sampler.get_alphas())
  //     std::cout << phi.transpose() << std::endl
  //               << std::endl;

  std::cout << "beta: " << sampler.get_beta().transpose() << std::endl;

  std::cout << "gamma: " << sampler.get_gamma().transpose() << std::endl;

  std::cout << "sigma: \n" << sampler.get_sigma() << std::endl;

  // std::vector<MatrixXd> preds = sampler.predict_one(
  //     fixed_covs[1], long_covs[1], orig_data[1].row(0).transpose(),
  //     chains);

  // MatrixXd mean = MatrixXd::Zero(preds[0].rows(), preds[0].cols());
  // for (int r=0; r < preds.size(); r++)
  //     mean += preds[r];

  // mean = mean.array() / preds.size();

  // std::cout << "true\n" << orig_data[1] << std::endl;
  // std::cout << "pred\n"
  //           << mean << std::endl;
}