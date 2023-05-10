#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <Eigen/Dense>
#include <deque>
#include <stan/math/prim.hpp>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>

#include "../protos/cpp/state.pb.h"
#include "PolyaGammaHybrid.h"
#include "precmat.hpp"
#include "rng.hpp"
#include "utils.hpp"

#include "base_sampler.hpp"

using namespace Eigen;

class Sampler: public BaseSampler {
 protected:
  std::vector<VectorXd> phis_vec;
  std::vector<VectorXd> alphas;

  VectorXd phi_0;
  MatrixXd v_0, v_0_inv;
  VectorXd phi_00;
  MatrixXd v_00;
  double lambda;
  double tau_0;
  VectorXd mu_alpha;
  MatrixXd sigma_alpha, sigma_alpha_inv;
  int iter_after_adapt = 1000;
  PolyaGammaHybridDouble* pg_rng = nullptr;

  std::string weightsmodel;

 public:
  Sampler() {}
  ~Sampler() {}

  Sampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXi> is_missing,
          std::vector<MatrixXd> long_covs, std::vector<VectorXd> fixed_covs,
          std::string weightsmodel = "LSB");

  Sampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXd> long_covs,
          std::vector<VectorXd> fixed_covs, std::string weightsmodel = "LSB");

  void set_base_measure(Eigen::MatrixXd phi00_ = Eigen::MatrixXd(0, 0),
                        Eigen::MatrixXd v00_ = Eigen::MatrixXd(0, 0),
                        double lambda_ = 1.0, double tau0_ = 1.0);

  void set_prior_lsb(Eigen::VectorXd mu_alpha_ = Eigen::VectorXd(0),
                     double var = 1.0);

  void initialize() override;

  void step() override;

  void sample_phis();

  void sample_pg();

  void relabel();

  Eigen::MatrixXd get_phi(int cluster_idx, Eigen::VectorXd fixed_cov=Eigen::VectorXd(0)) override {
    Eigen::MatrixXd phi = Map<MatrixXd>(phis_vec[cluster_idx].data(), rdim, rdim);
    return phi;
  }

  void set_cluster(const std::vector<int> clus) { this->clus_allocs = clus; }

  void sample_hypers();

  void simulate_missing();

  VectorXd lsb_weights(VectorXd covs, bool log_ = true, bool train_ = false);

  VectorXd get_beta() const { return beta_vec; }

  VectorXd get_gamma() const { return gamma_vec; }

  MatrixXd get_sigma() const { return sigma; }

  std::vector<VectorXd> get_phis() const { return phis_vec; }

  std::vector<VectorXd> get_alphas() const { return alphas; }

  std::vector<int> get_cluster() const { return clus_allocs; }

  void get_state_as_proto(State* out) override;

  void restore_from_proto(State state) override;

  std::vector<MatrixXd> predict_phi(VectorXd fixed_covs,
                                    std::deque<State> chains);

  std::vector<MatrixXd> predict_one(VectorXd fixed_covs, MatrixXd long_covs,
                                    VectorXd start, std::deque<State> chains);

  std::vector<MatrixXd> predict_one_step(VectorXd fixed_covs,
                                         MatrixXd long_covs, MatrixXd vals,
                                         std::deque<State> chains);

  std::vector<MatrixXd> predict_in_sample(int idx, int nsteps,
                                          VectorXd fixed_covs,
                                          MatrixXd long_covs, MatrixXd vals,
                                          std::deque<State> chains);

  VectorXd get_weights(VectorXd fixed_covs, bool log_ = true,
                       bool train_ = false) override;

  void set_clus_alloc(std::vector<int> clus_alloc);


  void set_phis(std::vector<MatrixXd>& phis) {
    for (int h = 0; h < phis.size(); h++) {
      VectorXd temp = Map<VectorXd>(phis[h].data(), phis[h].size());
      phis_vec[h] = temp;
    }
  }

  void set_sigma(const MatrixXd& sigma) {
    this->sigma = sigma;
    this->sigma_inv = PrecMat(stan::math::inverse_spd(sigma));
  }

  void set_beta(const MatrixXd& beta) { beta_vec = vectorize(beta); }

  void set_gamma(const MatrixXd& gamma) { gamma_vec = vectorize(gamma); }

  std::vector<MatrixXd> get_imputed_data() const { return imputed_data; }
};

#endif