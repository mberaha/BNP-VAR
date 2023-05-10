#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <deque>
#include <string>

#include "base_sampler.hpp"
#include "linear_ddp_sampler.hpp"
#include "sampler.hpp"


namespace py = pybind11;

std::shared_ptr<BaseSampler> create_sampler(
    int H, std::vector<MatrixXd> data, std::vector<MatrixXi> is_missing,
    std::vector<MatrixXd> long_covs, std::vector<VectorXd> fixed_covs,
    std::string model) {

  bool missing_data = (is_missing.size() > 0);
  if (missing_data) {
    if (model == "LinearDDP") {
      return std::make_shared<LinearDDPSampler>(H, data, is_missing, long_covs,
                                         fixed_covs);
    } else {
      return std::make_shared<Sampler>(H, data, is_missing, long_covs,
                                      fixed_covs, model);
    }
  } else {
    if (model == "LinearDDP") {
      return std::make_shared<LinearDDPSampler>(H, data, long_covs, fixed_covs);
    } else {
      return std::make_shared<Sampler>(H, data, long_covs, fixed_covs, model);
    }
  }
}


std::deque<py::bytes> run_mcmc(
    int H, int adapt, int burnin, int niter, int thin,
    const std::vector<MatrixXd> &data, const std::vector<MatrixXd> &long_covs,
    std::vector<VectorXd> &fixed_covs, const std::vector<MatrixXi> &missing,
    std::string model = "LSB", Eigen::MatrixXd phi00_ = Eigen::MatrixXd(0, 0),
    Eigen::MatrixXd v00_ = Eigen::MatrixXd(0, 0), double lambda_ = 1.0,
    double tau0_ = 1.0, Eigen::MatrixXd sigma0_ = Eigen::MatrixXd(0, 0),
    double nu_ = 0, Eigen::MatrixXd beta0 = Eigen::MatrixXd(0, 0),
    double varb = 1.0, Eigen::MatrixXd gamma0 = Eigen::MatrixXd(0, 0),
    double varg = 1.0, Eigen::VectorXd mu_alpha_ = Eigen::VectorXd(0),
    double vara = 0.5, Eigen::VectorXd linddp_mean = Eigen::VectorXd(0), 
    Eigen::MatrixXd linddp_var = Eigen::MatrixXd(0, 0)) {
  std::cout << "start" << std::endl;
  int log_every = 200;
  std::deque<py::bytes> out;

  std::cout << "H: " << H << std::endl;

  std::cout << "creating_sampler" << std::endl;
  std::shared_ptr<BaseSampler> sampler = create_sampler(H, data, missing,
                                                        long_covs, fixed_covs,
                                                        model);
  std::cout << "done" << std::endl;

  std::cout << "phi00_ \n" << phi00_ << std::endl;
  std::cout << "v00_ \n" << v00_ << std::endl;
  std::cout << "sigma0_ \n" << sigma0_ << std::endl;
  std::cout << "beta0 \n" << beta0 << std::endl;
  std::cout << "gamma0 \n" << gamma0 << std::endl;
  std::cout << "lambda: " << lambda_ << ", tau0: " << tau0_ << ", nu: " << nu_
            << std::endl;

  sampler->set_prior_sigma(sigma0_, nu_);
  sampler->set_prior_beta(beta0, varb);
  sampler->set_prior_gamma(gamma0, varg);
  sampler->set_prior_dp();

  if (model == "LinearDDP") {
    std::dynamic_pointer_cast<LinearDDPSampler>(sampler)->set_base_measure(
      linddp_mean, linddp_var);
  } else {
    std::dynamic_pointer_cast<Sampler>(sampler)->set_base_measure(phi00_, v00_, lambda_, tau0_);
    std::dynamic_pointer_cast<Sampler>(sampler)->set_prior_lsb(mu_alpha_, vara);
  }
  std::cout << "Before initializing" << std::endl;

  sampler->initialize();
  std::cout << "initialize_done" << std::endl;

  if (adapt > 0) {
    sampler->set_adapt(true);
    for (int i = 0; i < adapt; i++) {
      sampler->step();
      if ((i + 1) % log_every == 0) {
        py::print("Adapt, iter #", i + 1, " / ", adapt);
      }
      if ((i + 1) % 2000 == 0) {
        std::cout << "clus allocs: " << std::endl;

        for (auto c : sampler->get_clus_allocs()) std::cout << c << ", ";
        std::cout << std::endl;
      }
    }
    std::string s;
    State curr;
    sampler->get_state_as_proto(&curr);
    curr.SerializeToString(&s);
    out.push_back((py::bytes)s);
  }

  sampler->set_adapt(false);
  for (int i = 0; i < burnin; i++) {
    sampler->step();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }

    if ((i + 1) % 2000 == 0) {
      std::cout << "clus allocs: " << std::endl;
      for (auto c : sampler->get_clus_allocs()) std::cout << c << ", ";
      std::cout << std::endl;
    }
  }

  for (int i = 0; i < niter; i++) {
    sampler->step();
    if (i % thin == 0) {
      std::string s;
      State curr;
      sampler->get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
    if ((i + 1) % 2000 == 0) {
      std::cout << "clus allocs: " << std::endl;

      for (auto c : sampler->get_clus_allocs()) std::cout << c << ", ";
      std::cout << std::endl;
    }
  }

  std::string s;
  State curr;
  sampler->get_state_as_proto(&curr);
  curr.SerializeToString(&s);
  out.push_back((py::bytes)s);

  std::cout << "Final Gamma: \n" << sampler->get_gamma_mat() << std::endl;
  std::cout << "end" << std::endl;

  return out;
}

std::vector<MatrixXd> sample_phi_predictive(
    const VectorXd &fixed_covs,
    const std::vector<std::string> &serialized_chains) {
  std::deque<State> chains;
  for (int i = 0; i < serialized_chains.size(); i++) {
    State state;
    state.ParseFromString(serialized_chains[i]);
    chains.push_back(state);
  }

  Sampler sampler;

  return sampler.predict_phi(fixed_covs, chains);
}

std::vector<MatrixXd> sample_one_predictive(
    const MatrixXd &long_covs, const VectorXd &fixed_covs,
    const VectorXd &start, const std::vector<std::string> &serialized_chains) {
  std::deque<State> chains;
  for (int i = 0; i < serialized_chains.size(); i++) {
    State state;
    state.ParseFromString(serialized_chains[i]);
    chains.push_back(state);
  }

  Sampler sampler;

  return sampler.predict_one(fixed_covs, long_covs, start, chains);
}

std::vector<MatrixXd> sample_predictive_onestep(
    const MatrixXd &long_covs, const VectorXd &fixed_covs, const MatrixXd &vals,
    const std::vector<std::string> &serialized_chains) {
  std::deque<State> chains;
  for (int i = 0; i < serialized_chains.size(); i++) {
    State state;
    state.ParseFromString(serialized_chains[i]);
    chains.push_back(state);
  }

  Sampler sampler;
  return sampler.predict_one_step(fixed_covs, long_covs, vals, chains);
}

std::vector<MatrixXd> sample_predictive_insample(
    int idx, int nsteps, MatrixXd vals, MatrixXd long_covs, VectorXd fixed_covs,
    const std::vector<std::string> &serialized_chains) {
  std::deque<State> chains;
  for (int i = 0; i < serialized_chains.size(); i++) {
    State state;
    state.ParseFromString(serialized_chains[i]);
    chains.push_back(state);
  }

  Sampler sampler;
  return sampler.predict_in_sample(idx, nsteps, fixed_covs, long_covs, vals,
                                   chains);
}

PYBIND11_MODULE(pp_mix_cpp, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def("run_mcmc", &run_mcmc, "aaa");

  // m.def("run_mcmc_with_inits", &run_mcmc_with_inits, "aaa");

  m.def("sample_one_predictive", &sample_one_predictive, "aaa");

  m.def("sample_predictive_onestep", &sample_predictive_onestep, "aaa");

  m.def("sample_predictive_insample", &sample_predictive_insample, "aaa");

  m.def("sample_phi_predictive", &sample_phi_predictive, "aaa");
}