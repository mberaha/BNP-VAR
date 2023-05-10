#include "utils.hpp"

void to_proto(const MatrixXd &mat, EigenMatrix *out)
{
    out->set_rows(mat.rows());
    out->set_cols(mat.cols());
    *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

void to_proto(const VectorXd &vec, EigenVector *out)
{
    out->set_size(vec.size());
    *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

VectorXd to_eigen(const EigenVector &vec)
{
    int size = vec.size();
    Eigen::VectorXd out;
    if (size > 0)
    {
        const double *p = &(vec.data())[0];
        out = Map<const VectorXd>(p, size);
    }
    return out;
}

MatrixXd to_eigen(const EigenMatrix &mat)
{
    int nrow = mat.rows();
    int ncol = mat.cols();
    Eigen::MatrixXd out;
    if (nrow > 0 & ncol > 0)
    {
        const double *p = &(mat.data())[0];
        out = Map<const MatrixXd>(p, nrow, ncol);
    }
    return out;
}

double multi_normal_prec_lpdf(const VectorXd &x, const VectorXd &mu,
                                const PrecMat &sigma) {
  double out = sigma.get_log_det() * x.size();
  out -= ((x - mu).transpose() * sigma.get_cho_factor_eval()).squaredNorm();
  return 0.5 * out;
}

double multi_normal_prec_lpdf(const std::vector<VectorXd> &x,
                                const VectorXd &mu, const PrecMat &sigma) {
  int n = x.size();
  double out = sigma.get_log_det() * n;

  const MatrixXd &cho_sigma = sigma.get_cho_factor_eval();

  std::vector<double> loglikes(n);
  for (int i = 0; i < n; i++) {
    loglikes[i] = ((x[i] - mu).transpose() * cho_sigma).squaredNorm();
  }

  out -= std::accumulate(loglikes.begin(), loglikes.end(), 0.0);

  return 0.5 * out;
}