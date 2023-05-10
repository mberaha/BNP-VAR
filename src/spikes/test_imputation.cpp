#include <Eigen/Dense>
#include <iostream>
#include <stan/math/prim.hpp>
#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;

MatrixXd get_permutation_matrix(MatrixXi missing)
{
    VectorXi missing_vec = vectorize(missing);
    MatrixXd out(missing_vec.size(), missing_vec.size());
    std::vector<int> to_move;
    std::vector<int> leftovers;
    for (int i = 0; i < missing_vec.size(); i++)
    {
        if (missing_vec[i] > 0)
            to_move.push_back(i);
        else
            leftovers.push_back(i);
    }

    for (int i = 0; i < to_move.size(); i++)
        out(i, to_move[i]) = 1.0;

    for (int i = 0; i < leftovers.size(); i++)
        out(i + to_move.size(), leftovers[i]) = 1.0;

    return out;
}

MatrixXd impute_missing(
    Eigen::MatrixXd data, Eigen::MatrixXi missing, 
    Eigen::MatrixXd phi, Eigen::MatrixXd beta_mat,
    Eigen::MatrixXd gamma_mat, Eigen::MatrixXd long_covs,
    Eigen::VectorXd fixed_cov, Eigen::MatrixXd sigma_inv)
{
    using RowMat = Matrix<double, Dynamic, Dynamic, RowMajor>;
    int rdim = 3;
    MatrixXd id_r = MatrixXd::Identity(rdim, rdim);
    int l = data.rows() - 1;
    MatrixXd perm_mat = get_permutation_matrix(
        missing.block(1, 0, l, rdim));

    MatrixXd prec_mat = MatrixXd::Zero(l * rdim, l * rdim);
    VectorXd mu = VectorXd::Zero(l * rdim);
    mu.head(rdim) = phi * data.row(0).transpose() +
                    beta_mat * long_covs.row(1).transpose() +
                    gamma_mat * fixed_cov;

    for (int i = 1; i < l; i++)
    {
        VectorXd prev = mu.segment((i - 1) * rdim, rdim);
        VectorXd curr = phi * prev +
                        beta_mat * long_covs.row(i + 1).transpose() +
                        gamma_mat * fixed_cov;
        mu.segment(i * rdim, rdim) = curr;
            
    }

    MatrixXd diag_elem = (id_r + phi).transpose() * sigma_inv * (id_r + phi);
    for (int i = 0; i < l - 1; i++)
    {
        prec_mat.block(i * rdim, i * rdim, rdim, rdim) = diag_elem;
        prec_mat.block(i * rdim, (i + 1) * rdim, rdim, rdim) = \
            phi.transpose() * sigma_inv;
        prec_mat.block((i + 1) * rdim, i * rdim, rdim, rdim) = \
            phi.transpose() * sigma_inv;
    }
    prec_mat.block((l - 1) * rdim, (l - 1) * rdim, rdim, rdim) = sigma_inv;

    prec_mat = perm_mat * prec_mat * perm_mat.transpose();
    mu = perm_mat * mu;

    VectorXd data_vect = vectorize(data);

    data_vect = data_vect.tail(data_vect.size() - rdim);
    data_vect = perm_mat * data_vect;

    int k = missing.sum();

    MatrixXd cov_missing = prec_mat.block(0, 0, k, k).inverse();
    VectorXd mean_missing = mu.head(k) -
        cov_missing * prec_mat.block(0, k, k, data_vect.size() - k) * (
            data_vect.tail(data_vect.size() - k) - 
            mu.tail(data_vect.size() - k));

    VectorXd sampled = stan::math::multi_normal_rng(
        mean_missing, cov_missing, Rng::Instance().get());

    data_vect.head(k) = sampled;

    MatrixXd out(l+1, rdim);
    out.row(0) = data.row(0);
    data_vect = perm_mat.transpose() * data_vect;

    out.block(1, 0, l, rdim) =  Map<RowMat>(data_vect.data(), l, rdim);

    return out;
}


int main() {
    MatrixXd beta = MatrixXd::Identity(3, 3);
    MatrixXd gamma = MatrixXd::Ones(3, 2);
    MatrixXd sigma_true = MatrixXd::Identity(3, 3).array() * 0.25;
    MatrixXd phi = MatrixXd::Identity(3, 3);
    phi << 0.5, 1.0, 0.3, 0.0, 1.0, 0.2, 1.0, 0.0, 0.2;
    

    VectorXd mean_fixed = VectorXd::Zero(2);
    MatrixXd cov_fixed = MatrixXd::Identity(2, 2).array();

    VectorXd mean_long = VectorXd::Zero(3);
    MatrixXd cov_long = MatrixXd::Identity(3, 3);

    VectorXd fixed_cov = stan::math::multi_normal_rng(
        mean_fixed, cov_fixed, Rng::Instance().get());

    MatrixXd data(10, 3);
    MatrixXd long_covs(10, 3);
    data.row(0) = stan::math::multi_normal_rng(
        mean_long, cov_long, Rng::Instance().get());
    for (int j=1; j < 10; j++) {
        long_covs.row(j) = stan::math::multi_normal_rng(
            mean_long, cov_long, Rng::Instance().get());
        VectorXd mean = beta * long_covs.row(j).transpose() +
                        gamma * fixed_cov;

        mean += phi * data.row(j - 1).transpose();
        data.row(j) = stan::math::multi_normal_rng(
                                  mean, sigma_true, Rng::Instance().get())
                                  .transpose();
    }

    MatrixXi missing = MatrixXi::Zero(10, 3);
    missing(2, 2) = 1;
    missing(3, 2) = 1;
    missing(5, 1) = 1;

    std::cout << "True data: " << data(2, 2) << ", " << data(3, 2) <<
                 ", " << data(5, 1) << std::endl;

    MatrixXd imputed;
    for (int k=0; k < 100; k++) {
        imputed = impute_missing(data, missing, phi, beta, gamma,
                                  long_covs, fixed_cov, sigma_true.inverse());
    }
    
    // std::cout << "orig data\n" << data << "\nImputed\n" << imputed << std::endl;

    std::cout << "Imputed data: " << imputed(2, 2) << ", " << imputed(3, 2)
              << ", " << imputed(5, 1) << std::endl;
}
