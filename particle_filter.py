from scipy.spatial.transform import Rotation as R
import numpy as np
from lidar_2d import lidar2DLikelihood, Lidar2DMeasure, LidarInfo
import time
import logging

def _assert_position_size(ps, particle_num):
    if len(ps) != particle_num:
        raise ValueError("Expected number of rotations to be equal to "
                            "number of timestamps given, got {} positions "
                            "and {} particles."
                            .format(len(ps), particle_num))

def _assert_rotation_size(rs, particle_num):
    if len(rs) != particle_num:
        raise ValueError("Expected number of rotations to be equal to "
                            "number of timestamps given, got {} rotations "
                            "and {} particles."
                            .format(len(rs), particle_num))                        

def _assert_weight_size(ws, particle_num):
    if len(ws) != particle_num:
        raise ValueError("Expected number of rotations to be equal to "
                            "number of timestamps given, got {} weights "
                            "and {} particles."
                            .format(len(ws), particle_num)) 


def resample(ps, rs, weights, particle_num):
    """
    Parameters
    ----------
    ps : array_like, shape (N, 2)
        position of all particles
    rs : `Rotation` instance, shape (N,)
        rotation of all particles, Must contain N rotations
    weights: weight vector, shape (N, )
    particle_num: particle number, N

    Returns
    -------
    This function return ps, rs, weights
    """

    logging.info("do resample!")

    _assert_position_size(ps, particle_num)
    _assert_rotation_size(rs, particle_num)
    _assert_weight_size(weights, particle_num)

    resample_anchor = np.arange(0.0, 1.0, 1 / particle_num) + np.random.random() / particle_num
    weight_cum_sum = np.cumsum(weights)

    indices = []
    wid = 0
    for anchor in resample_anchor:
        while anchor > weight_cum_sum[wid]:
            wid += 1
        # get index when anchor < bin boundary
        indices.append(wid)

    if len(indices) != particle_num:
        raise ValueError("len(indices) != particle_num")
    
    logging.info("preserved particle: {}".format(list(dict.fromkeys(indices))))
    r_quat = rs.as_quat()
    

    return ps[indices], R.from_quat(r_quat[indices]), np.ones(particle_num) / particle_num



class Lidar2DParticleFilter(object):
    def __init__(self, poly_map, particle_num=1024, lidar_info=LidarInfo()):
        
        self.particle_num = particle_num
        self.lidar_info = lidar_info
        self.map = poly_map
        
        self.ps = np.zeros([self.particle_num, 2])
        rotvecs = np.zeros([self.particle_num, 3])
        self.rs = R.from_rotvec(rotvecs)
        self.weights = np.ones(self.particle_num) / self.particle_num

        self.ps_viz = self.ps

        self.est_cov_p = np.identity(2) *1e-1
        self.est_cov_r = np.identity(1) *1e-1


    def setParticleState(self, ps, rs):
        _assert_position_size(ps, self.particle_num)
        _assert_rotation_size(rs, self.particle_num)
        self.ps = ps
        self.rs = rs
        self.ps_viz = self.ps

    def positionViz(self, scale=10.):
        zero_mean = self.ps_viz - self.est_p
        zero_mean *= scale
        return (zero_mean + self.est_p)

    def update(self, dev_measure_real, p_inc, r_inc):
        # update particles
        self.ps += self.rs.apply(np.array([p_inc[0], p_inc[1], 0]))[:,:2]
        self.rs = r_inc * self.rs
        
        # sample particles
        self.ps += np.random.multivariate_normal([0, 0], self.est_cov_p * 1e1, self.particle_num)
        noise_rotvec = np.zeros([self.particle_num, 3])
        noise_rotvec[:, -1:] = np.random.multivariate_normal([0], self.est_cov_r * 1e1 , self.particle_num)
        self.rs = R.from_rotvec(noise_rotvec) * self.rs

        self.ps_viz = self.ps

        # calculate likelihood
        dev_measure_samples = Lidar2DMeasure(self.ps, self.rs, self.lidar_info, self.map)
        # print(lidar_segments)
        log_likelihood = lidar2DLikelihood(dev_measure_real, dev_measure_samples, self.lidar_info, self.particle_num)
        log_likelihood = log_likelihood.get()
        # print(log_likelihood)

        logging.info("post process")
        # update weights
        self.weights *= np.exp(log_likelihood)
        # print(self.weights)
        self.weights /= np.sum(self.weights)

        # update mean
        self.est_p = self.ps.T.dot(self.weights)
        self.est_r = self.rs.mean(self.weights)

        # update covariance
        res_p = self.ps - self.est_p
        res_rv = (self.rs * self.est_r.inv()).as_rotvec()

        self.est_cov_p = np.dot(self.weights * res_p.T, res_p)
        self.est_cov_r = np.dot(self.weights * res_rv.T, res_rv)[2:,2:]

        logging.info("est_cov_p: {}, est_cov_r: {}".format(self.est_cov_p, self.est_cov_r))

        best_likelihood = max(log_likelihood)
        # print("best_likelihood: {}".format(best_likelihood))
        logging.info("best_likelihood: {}".format(best_likelihood))
        # print("post process cost: {}".format(time.time() - t0))        

        n_eff = 1./self.weights.dot(self.weights)
        n_th = self.particle_num / 2
        if n_eff > n_th:
            return

        t0 = time.time()
        self.ps, self.rs, self.weights = resample(self.ps, self.rs, self.weights, self.particle_num)
        logging.info("resample finished")
        # print("resample cost: {}".format(time.time() - t0))