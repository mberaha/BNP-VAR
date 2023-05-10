#ifndef LINEAR_DDP_SAMPLER_HPP
#define LINEAR_DDP_SAMPLER_HPP

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


class LinearDDPSampler: public BaseSampler {
 protected:
    std::vector<VectorXd> cluster_regressors;

    VectorXd cluster_reg_mean;
    MatrixXd cluster_reg_cov;

    Eigen::MatrixXd mh_prop_cov;
    Eigen::VectorXd mh_rolling_mean;

    int nburn_iter;
    double adapt_decay;

 public:
   LinearDDPSampler() {}

   ~LinearDDPSampler() {};

   LinearDDPSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXi> is_missing,
          std::vector<MatrixXd> long_covs, std::vector<VectorXd> fixed_covs) :
         BaseSampler(H, data, is_missing, long_covs, fixed_covs) {};

    LinearDDPSampler(int H, std::vector<MatrixXd> data, std::vector<MatrixXd> long_covs,
          std::vector<VectorXd> fixed_covs): BaseSampler(H, data, long_covs, fixed_covs) {};

   void set_base_measure(VectorXd beta0, MatrixXd cov);

   void set_prior_dp(double totalmass = 1.0) {dp_mass = totalmass;}

   void initialize() override;

   void step() override;

   void sample_cluster_regressors();

   Eigen::MatrixXd get_phi(int cluster_idx, Eigen::VectorXd fixed_cov) override {
       Eigen::MatrixXd clus_reg = Map<MatrixXd>(
           cluster_regressors[cluster_idx].data(), rdim * rdim, fdim);
       Eigen::VectorXd phi = clus_reg * fixed_cov;
       Eigen::MatrixXd phimat = Map<MatrixXd>(phi.data(), rdim, rdim);
       return phimat;
   }

    Eigen::MatrixXd get_phi(Eigen::VectorXd regressor, Eigen::VectorXd fixed_cov) {
       Eigen::MatrixXd clus_reg = Map<MatrixXd>(
           regressor.data(), rdim * rdim, fdim);
       Eigen::VectorXd phi = clus_reg * fixed_cov;
       Eigen::MatrixXd phimat = Map<MatrixXd>(phi.data(), rdim, rdim);
       return phimat;
   }



   void simulate_missing();

   void get_state_as_proto(State* out) override;

  void restore_from_proto(State state) override;

  VectorXd get_weights(VectorXd fixed_covs, bool log_ = true,
                       bool train_ = false) override {
    Eigen::VectorXd out;
    if (log_)
      out = log(dp_weights.array());
    else
      out = dp_weights;
    
    return out;
    }

    double adapt_step(int iter) {
        if (iter < nburn_iter) {
            return std::exp(- adapt_decay * iter);
        } else {
            return 0;
        }
    }

    void set_adaptation(int nburn) {
        nburn_iter = nburn;
        adapt_decay = -2.0 * std::log(0.1) / (1.0 * nburn);
    }


};

#endif