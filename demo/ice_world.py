from polygon_map import PolygonMap
from lidar_2d import *
from polygon_map import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from particle_filter import Lidar2DParticleFilter
import matplotlib
from motion import Motion

if __name__ == "__main__":


    matplotlib.use('TkAgg')
    poly_map = PolygonMap('ice_world')
    motion = Motion('ice_world_0')

    fps = 20
    p_observe_noise = 0.05
    r_observe_noise = 0.05

    lidar_info = LidarInfo(resolution=32, distance=50, std_dev=1e-1, fov=[-0.5, 0.5])
    pf = Lidar2DParticleFilter(poly_map, particle_num=512, lidar_info=lidar_info, p_cov=0.1, r_cov=0.1)
    
    sim_time = motion.t_range[1] - motion.t_range[0]
    ts = np.linspace(*motion.t_range, int(fps * sim_time))

    lidar_pts = motion.spline.lin_p(0)[:2]
    lidar_rs = motion.spline.ang_p(0)

    pf.setParticleState(
        np.array( motion.spline.lin_p([0] * pf.particle_num)[:, :2]), 
        motion.spline.ang_p([0] * pf.particle_num))

    last_t = ts[0]

    for t in ts:
        lidar_pts = motion.spline.lin_p(t)[:2]
        lidar_rs = motion.spline.ang_p(t)

        # odometry observe
        lidar_pts_inc  = lidar_pts - motion.spline.lin_p(last_t)[:2] + np.random.random(2) * p_observe_noise
        lidar_rs_inc = R.from_rotvec([0,0, np.random.random(1) * r_observe_noise]) * lidar_rs * motion.spline.ang_p(last_t).inv()
        last_t = t

        # sensor observe
        lidar_measure = Lidar2DMeasure(lidar_pts, lidar_rs, lidar_info, poly_map)
        
        pf.update(lidar_measure, lidar_pts_inc, lidar_rs_inc)
        beam_viz = createLidarSegments(np.array([pf.est_p]), R.from_quat([pf.est_r.as_quat()]) , lidar_info, lidar_measure.get())[::1]

        if 1:
            plt.clf()
            for seg in poly_map.segments:
                plt.plot(seg[:, 0], seg[:, 1], 'k')
            for seg in beam_viz:
                plt.plot(seg[:, 0], seg[:, 1], 'r', lw=0.5)
            plt.plot(*pf.ps_viz.T, '.', markersize=1, alpha=0.3)
            plt.plot(*motion.spline.lin_p(ts)[:,:2].T, 'g--')
            
            plt.axis([-5, 50, -5, 50])
            plt.gca().set_aspect('equal', adjustable='box')
            
            plt.pause(0.001)
            