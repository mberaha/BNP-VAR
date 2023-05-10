#include <Eigen/Dense>
#include <deque>
#include <stan/math/prim.hpp>
#include <vector>

#include "../rng.hpp"
#include "../sampler.hpp"
#include "../PolyaGammaHybrid.h"

using namespace Eigen;
using namespace stan;

void sample_pg(int H, std::vector<int> clus_allocs,
               std::vector<VectorXd> fixed_covs) {
  // std::cout << "sample pg" << std::endl;
  int ndata = clus_allocs.size();
  int fdim = fixed_covs[0].size();
  VectorXd mu_alpha = VectorXd::Zero(fdim);
  MatrixXd sigma_alpha = MatrixXd::Identity(fdim, fdim).array() * 3.0;
  MatrixXd sigma_alpha_inv = stan::math::inverse_spd(sigma_alpha);
  PolyaGammaHybridDouble* pg_rng = new PolyaGammaHybridDouble(12313123);
  std::vector<VectorXd> alphas(H);


  for (int h = 0; h < H - 1; h++) {
    alphas[h] = mu_alpha;
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
        double c = fixed_covs[i].transpose() * alphas[h];
        omegas(pos, pos) = pg_rng->draw(1, c);
        curr_covs.row(pos) = fixed_covs[i].transpose();
        if (clus_allocs[i] == h)
          kappa[pos] = 1.0;
        else
          kappa[pos] = 0.0;
        pos += 1;
      }
    }

    MatrixXd sigma_alpha_post =
        curr_covs.transpose() * omegas * curr_covs + sigma_alpha_inv;

    VectorXd mu_alpha_post = sigma_alpha_post.ldlt().solve(
        curr_covs.transpose() * kappa + sigma_alpha_inv * mu_alpha);

    std::cout << "mu_alpha_post: " << mu_alpha_post.transpose() << std::endl;
  }
}

int main() {
  MatrixXd beta_true = MatrixXd::Identity(3, 3);
  MatrixXd gamma_true = MatrixXd::Zero(3, 2);
  // MatrixXd gamma_true(3, 2);
  // gamma_true << 0, 1, 0.5, 1, 0, 0;
  MatrixXd sigma_true = MatrixXd::Identity(3, 3).array() * 0.25;
  VectorXd alpha1(2);
  alpha1 << 2.0, 2.0;
  VectorXd alpha2(2);
  alpha2 << -2.0, -2.0;
  int ndata = 100;
  int ts = 10;
  std::vector<MatrixXd> orig_data(ndata);
  std::vector<MatrixXd> long_covs(ndata);
  std::vector<VectorXd> fixed_covs(ndata);
  std::vector<MatrixXi> is_missing(ndata);

  VectorXd mean_fixed = VectorXd::Zero(2);
  MatrixXd cov_fixed = MatrixXd::Identity(2, 2).array();


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
  }

  int H = 4;
  sample_pg(H, true_clus, fixed_covs);

}