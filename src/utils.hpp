#ifndef UTILS_HPP
#define UTILS_HPP

#include "../protos/cpp/state.pb.h"
#include "precmat.hpp"
#include <numeric>
#include <Eigen/Dense>


using namespace Eigen;

void to_proto(const MatrixXd &mat, EigenMatrix *out);

void to_proto(const VectorXd &vec, EigenVector *out);

Eigen::VectorXd to_eigen(const EigenVector &vec);

Eigen::MatrixXd to_eigen(const EigenMatrix &vec);

template<typename t>
Matrix<t, Dynamic, 1> vectorize(const Matrix<t, Dynamic, Dynamic>& mat)
{
    Matrix<t, Dynamic, 1> out(mat.size());
    int nrow = mat.rows();
    int ncol = mat.cols();
    for (int i = 0; i < nrow; i++) {
        out.block(i * ncol, 0, ncol, 1) = mat.row(i).transpose();
    }
    return out;
}

/*
 * Evaluates the log likelihood for the model
 *      X ~ N(\mu, \sigma^{-1})
 */
double multi_normal_prec_lpdf(const VectorXd &x, const VectorXd &mu,
                              const PrecMat &sigma);

/*
 * Evaluates the log likelihood for the model
 *      X_1, \ldots, X_n ~ N(\mu, \sigma^{-1})
 */
double multi_normal_prec_lpdf(const std::vector<VectorXd> &x,
                              const VectorXd &mu, const PrecMat &sigma);


#endif