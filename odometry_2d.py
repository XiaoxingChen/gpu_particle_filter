import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class Odometry2D(object):
    def __init__(self, p_cov=np.array([0.05, 0.02]), r_cov=0.1):
        self.p_cov = p_cov
        self.r_cov = r_cov

        self.prev_t = None
        self.prev_p = None
        self.prev_r = None

        self.this_t = None
        self.this_p = None
        self.this_r = None

    def update(self, t, p, r):
        self.prev_t = self.this_t
        self.prev_p = self.this_p
        self.prev_r = self.this_r

        self.this_t = t
        self.this_p = p
        self.this_r = r

    def latestObservation(self):
        inc_p = np.zeros(2)
        inc_r = R.identity() 

        if self.prev_p is not None and self.prev_r is not None:
            inc_p = self.prev_r.as_matrix().T[:2,:2].dot(self.this_p - self.prev_p)
            inc_p += (np.random.random(2)- 0.5) * self.p_cov
            inc_r = self.this_r * self.prev_r.inv()
            inc_r = R.from_rotvec([0, 0, (np.random.random(1) - 0.5) * self.r_cov]) * inc_r

        return inc_p, inc_r

    

