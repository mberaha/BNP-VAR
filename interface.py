import os
import sys
import numpy as np

from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32

from protos.py.state_pb2 import EigenMatrix, EigenVector, State

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import pp_mix_cpp  # noqa


def to_numpy(obj):
    if isinstance(obj, EigenMatrix):
        out = np.array(obj.data).reshape(obj.rows, obj.cols, order="F")
    elif isinstance(obj, EigenVector):
        out = np.array(obj.data)
    else:
        raise ValueError("incorrect object type")

    return out


def to_proto(array):
    if array.ndim == 1:
        out = EigenVector()
        out.size = len(array)
        out.data.extend(array.tolist())
    else:
        out = EigenMatrix()
        out.rows = array.shape[0]
        out.cols = array.shape[1]
        out.data.extend(array.reshape(1, -1, order='F').tolist()[0])
    return out


def loadChains(filename, msgType=State):
    out = []
    with open(filename, "rb") as fp:
        buf = fp.read()

    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n:n+msg_len]
        try:
            msg = msgType()
            msg.ParseFromString(msg_buf)
            out.append(msg)
            n += msg_len
        except Exception as e:
            break

    return out


def writeChains(chains, filename):
    with open(filename, "wb") as fp:
        for c in chains:
            try:
                msgStr = c.SerializeToString()
                delimiter = _VarintBytes(len(msgStr))
                fp.write(delimiter + msgStr)
            except Exception as e:
                print(e)
                break


def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


class Sampler(object):
    def __init__(self, H, model="LSB"):
        self.H = H
        self.model = model

    def run_mcmc(self, adapt, burnin, niter, thin, responses, long_covs,
                 fixed_covs, is_missing=[], start_cluster=[], start_phis=[],
                 start_sigma=[]):
        # self._serialized_chains = pp_mix_cpp.run_mcmc(
        #     self.H, burnin, niter, thin, responses, long_covs,
        #     fixed_covs, is_missing, start_cluster, start_phis,
        #     start_sigma)
        self.responses = responses
        self.long_covs = long_covs
        self.fixed_covs = fixed_covs

        self._serialized_chains = pp_mix_cpp.run_mcmc(
            self.H, adapt, burnin, niter, thin, responses, long_covs,
            fixed_covs, is_missing, self.model,
            self.phi00, self.v00, self.lamb, self.tau, self.sigma0,
            self.nu, self.beta0, self.varb, self.gamma0, self.varg,
            self.alpha0, self.vara, self.mean_reg, self.var_reg)

        self.chains = list(map(
            lambda x: getDeserialized(x, State), self._serialized_chains))

        return self.chains

    def set_prior(self, phi00=np.zeros((2, 2)), v00=np.eye(4), lamb=1.0, tau=1.0,
                  sigma0=np.eye(2), nu=0, beta0=np.zeros((2, 1)),
                  varb=1.0, gamma0=np.zeros((2, 11)), varg=1.0, alpha0=np.zeros(14), vara=0.5,
                  mean_reg=np.zeros(1), var_reg=np.zeros((1, 1))):
        self.phi00 = phi00
        self.v00 = v00
        self.lamb = lamb
        self.tau = tau
        self.sigma0 = sigma0
        self.nu = nu
        self.beta0 = beta0
        self.varb = varb
        self.gamma0 = gamma0
        self.varg = varg
        self.alpha0 = alpha0
        self.vara = vara
        self.mean_reg = mean_reg
        self.var_reg = var_reg

    def sample_predictive(self, long_covs, fixed_covs, start):
        return pp_mix_cpp.sample_one_predictive(
            long_covs, fixed_covs, start, self._serialized_chains)

    def sample_predictive_onestep(self, long_covs, fixed_covs, start):
        return pp_mix_cpp.sample_predictive_onestep(
            long_covs, fixed_covs, start, self._serialized_chains)

    def predict_insample(self, idx, long_covs, fixed_covs, nsteps):
        return pp_mix_cpp.sample_predictive_insample(
            idx, nsteps, self.responses[idx], long_covs, fixed_covs,
            self._serialized_chains)

    def sample_phi_predictive(self, fixed_covs):
        return pp_mix_cpp.sample_phi_predictive(
            fixed_covs, self._serialized_chains)
