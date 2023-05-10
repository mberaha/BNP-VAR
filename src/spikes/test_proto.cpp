#include "../../protos/cpp/state.pb.h"
#include "../utils.hpp"
#include <Eigen/Dense>

using namespace Eigen;

int main() {
    
    MatrixXd id = MatrixXd::Identity(2, 2);

    EigenMatrix a;
    
    a.set_rows(2);
    a.set_cols(2);


    *a.mutable_data() = {id.data(), id.data() + 4};

    a.PrintDebugString();

    State state;
    to_proto(id, state.mutable_beta());
    state.PrintDebugString();


}