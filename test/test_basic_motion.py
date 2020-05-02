from polygon_map import PolygonMap
from lidar_2d import *
from polygon_map import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from particle_filter import Lidar2DParticleFilter
import matplotlib

if __name__ == "__main__":

    if len(sys.argv) > 1:
        map_name = sys.argv[1]
    else:
        map_name = 'basic'

    matplotlib.use('TkAgg')

    map_lib = os.path.dirname(os.path.abspath(__file__)) + '/../map_lib.yaml'
    poly_map = PolygonMap(map_name)
    lidar_info = LidarInfo(resolution=32, distance=50, std_dev=1e-1, fov=[-0.5, 0.5])
    pf = Lidar2DParticleFilter(poly_map, particle_num=512, lidar_info=lidar_info)
    
    ts = np.linspace(0, 40, 200)

    lidar_pts = np.ones([1,2]) * 30 + np.array([[np.cos(ts[0]), np.sin(ts[0])]]) * 5
    lidar_rs = R.from_rotvec(np.array([[0, 0, 0.5 * ts[0]]]))

    pf.setParticleState(
        np.array([lidar_pts[0]] * pf.particle_num), 
        R.from_quat([lidar_rs.as_quat()[0]] * pf.particle_num))

    lidar_pts_last = lidar_pts
    lidar_rs_last = lidar_rs

    for t in ts:
        lidar_pts = np.ones([1,2]) * 30 + np.array([[np.cos(t), np.sin(t)]]) * 5
        lidar_rs = R.from_rotvec(np.array([[0, 0, -0.5 * t]]))

        lidar_pts_inc = (lidar_pts - lidar_pts_last) + np.random.random([pf.particle_num, 2]) * 0.05
        lidar_rs_inc = lidar_rs * lidar_rs_last.inv()

        lidar_pts_last = lidar_pts
        lidar_rs_last = lidar_rs

        lidar_measure = Lidar2DMeasure(lidar_pts, lidar_rs, lidar_info, poly_map)
        
        pf.update(lidar_measure, lidar_pts_inc, lidar_rs_inc)
        beam_viz = createLidarSegments(np.array([pf.est_p]), R.from_quat([pf.est_r.as_quat()]) , lidar_info, lidar_measure.get())[::1]

        if 1:
            plt.clf()
            for seg in poly_map.segments:
                plt.plot(seg[:, 0], seg[:, 1], 'k')
            for seg in beam_viz:
                plt.plot(seg[:, 0], seg[:, 1], 'r', lw=0.5)
            plt.plot(*pf.ps_viz.T, '.', markersize=1)
            plt.xlim(-5,60)
            plt.ylim(-5,60)
            # print(pf.ps)
            
            plt.pause(0.001)
            # plt.show()