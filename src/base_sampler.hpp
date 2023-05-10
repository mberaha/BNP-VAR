#ifndef BASESAMPLER_HPP
#define BASESAMPLER_HPP

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

using namespace Eigen;

class BaseSampler {
 protected:
  int rdim;
  int ldim;
  int fdim;
  int H;

  int ndata;
  int sum_time_steps;
  std::vector<MatrixXd> orig_data;
  std::vector<MatrixXd> imputed_data;
  std::vector<MatrixXi> is_missing;
  std::vector<MatrixXd> long_covs;
  std::vector<VectorXd> fixed_covs;

  VectorXd beta_vec;
  VectorXd gamma_vec;
  MatrixXd sigma;
  PrecMat sigma_inv;
  std::vector<MatrixXd> sigma_t;
  std::vector<PrecMat> sigma_inv_t;

  VectorXd weights;
  std::vector<int> clus_allocs;

  VectorXd mu_beta;
  MatrixXd sigma_beta, sigma_beta_inv;
  VectorXd mu_gamma;
  MatrixXd sigma_gamma, sigma_gamma_inv;
  double nu;
  MatrixXd sigma_0;

  Eigen::VectorXd dp_weights;
  double dp_mass;
  
  int seed = 12314312;

  bool missing_data = false;
  bool adapt = false;
  int iter = 0;

 public:
    BaseSampler() {}
    virtual ~BaseSampler() {};

    BaseSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXi> is_missing,
          std::vector<MatrixXd> long_covs, std::vector<VectorXd> fixed_covs);

    BaseSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXd> long_covs,
          std::vector<VectorXd> fixed_covs);

    void set_prior_sigma(Eigen::MatrixXd sigma0_ = Eigen::MatrixXd(0, 0),
                       double nu_ = 0);

    void set_prior_beta(Eigen::MatrixXd beta0 = Eigen::MatrixXd(0, 0),
                        double var = 1.0);

    void set_prior_gamma(Eigen::MatrixXd gamma0 = Eigen::MatrixXd(0, 0),
                       double var = 1.0);

    void set_prior_dp(double totalmass = 1.0) { dp_mass = totalmass; }

    virtual void initialize();

    virtual void step() = 0;

    void sample_beta();

    void sample_gamma();

    void sample_sigma();

    void sample_allocs();

    void sample_dp();

    // void simulate_missing();

    virtual void get_state_as_proto(State* out);

    virtual void restore_from_proto(State state);

    virtual Eigen::MatrixXd get_phi(
        int cluster_idx,  Eigen::VectorXd fixed_cov=Eigen::VectorXd(0)) = 0;

    virtual Eigen::VectorXd get_weights(VectorXd fixed_covs, bool log_ = true,
                       bool train_ = false) = 0;

    std::vector<int> get_clus_allocs() { return clus_allocs; }

    MatrixXd get_gamma_mat() {
        MatrixXd out = Map<MatrixXd>(gamma_vec.data(), rdim, fdim);
        return out;
    }
                       
    // virtual Eigen::MatrixXd sample_predictive_phi(
    //     VectorXd fixed_covs, State state, bool already_restored) = 0;

    // std::vector<MatrixXd> predict_phi(VectorXd fixed_covs,
    //                                 std::deque<State> chains);

    
    // std::vector<MatrixXd> predict_one(VectorXd fixed_covs,
    //                                        MatrixXd long_covs, VectorXd start,
    //                                        std::deque<State> chains);

    // std::vector<MatrixXd> predict_one_step(VectorXd fixed_covs,
    //                                             MatrixXd long_covs,
    //                                             MatrixXd vals,
    //                                             std::deque<State> chains);
                                        
    // std::vector<MatrixXd> predict_in_sample(int idx, int nsteps,
    //                                              VectorXd fixed_covs,
    //                                              MatrixXd long_covs,
    //                                              MatrixXd vals,
    //                                              std::deque<State> chains);



    MatrixXd get_permutation_matrix(MatrixXi missing);


  void set_adapt(const bool adapt_) {
    adapt = adapt_;
    if (adapt == false) {
      std::cout << "Finished adaptation, current clustering: " << std::endl;
      for (auto c : clus_allocs) std::cout << c << ", ";
      std::cout << std::endl;
    }
    iter = 0;
  }

};

#endif