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
import logging
from odometry_2d import Odometry2D

if __name__ == "__main__":
    map_name, traj_id = 'ice_world', '0'
    if len(sys.argv) > 1:
        map_name = sys.argv[1]
    if len(sys.argv) > 2:
        traj_id = sys.argv[2]

    matplotlib.use('TkAgg')
    poly_map = PolygonMap(map_name)
    motion = Motion('{}_{}'.format(map_name, traj_id))
    odom = Odometry2D()
    logging.basicConfig(filename=os.path.dirname(os.path.abspath(__file__)) + '/../__pycache__/gpu_pf.log', 
        format='[%(asctime)s] - %(message)s', level=logging.INFO, filemode='w')

    fps = 20

    lidar_info = LidarInfo(resolution=32, distance=50, std_dev=1e-1, fov=[-0.8, 0.8])
    pf = Lidar2DParticleFilter(poly_map, particle_num=512, lidar_info=lidar_info)

    logging.info(
        """Simulator Start! 
        Map: {}, trajectory ID: {},
        lidar_info: {},
        Particle num: {}
        """.format(map_name, traj_id, lidar_info, pf.particle_num))
    
    sim_time = motion.t_range[1] - motion.t_range[0]
    ts = np.linspace(*motion.t_range, int(fps * sim_time))

    lidar_position = motion.spline.lin_p(ts[0])[:2]
    lidar_rotation = motion.spline.ang_p(ts[0])

    pf.setParticleState(
        np.array( motion.spline.lin_p([ts[0]] * pf.particle_num)[:, :2]), 
        motion.spline.ang_p([ts[0]] * pf.particle_num))

    last_t = ts[0]
    real_traj = []
    fig = plt.figure(figsize=(6,6))

    while True:
        for t in ts:
            lidar_position = motion.spline.lin_p(t)[:2]
            lidar_rotation = motion.spline.ang_p(t)

            # odometry observe
            odom.update(t, lidar_position, lidar_rotation)

            # sensor observe
            logging.info("actual observe")
            lidar_measure = Lidar2DMeasure(lidar_position, lidar_rotation, lidar_info, poly_map)
            
            logging.info("update filter")
            # pf.update(lidar_measure, lidar_pts_inc, lidar_rs_inc)
            pf.update(lidar_measure, *odom.latestObservation())
            viz_measure = lidar_measure.get()
            logging.info("read measure done")
            viz_measure[viz_measure > lidar_info.distance] = 0
            beam_viz = createLidarSegments(np.array([pf.est_p]), R.from_quat([pf.est_r.as_quat()]) , lidar_info, viz_measure)[::1]
            real_traj.append(pf.est_p)
            if 1:
                plt.clf()
                for seg in poly_map.segments:
                    plt.plot(seg[:, 0], seg[:, 1], 'k')
                for seg in beam_viz:
                    plt.plot(seg[:, 0], seg[:, 1], 'r', lw=0.5)
                plt.plot(lidar_position[0], lidar_position[1], 'gx', markersize=15, label='actual')
                plt.plot(*pf.positionViz(scale=5).T, '.', markersize=3, alpha=0.5)
                plt.plot(*motion.spline.lin_p(ts)[:,:2].T, 'g--', label='actual')
                plt.plot(*np.array(real_traj).T, 'C5-', label='estimated')
                
                
                plt.axis([-5, 50, -5, 50])
                plt.legend()
                plt.gca().set_aspect('equal', adjustable='box')
                
                plt.pause(0.001)
            