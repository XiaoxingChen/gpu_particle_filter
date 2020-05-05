import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from cl_rocket import CLRocket
import pyopencl.array as cl_array
from polygon_map import PolygonMap
import matplotlib.pyplot as plt
import logging

def _assert_pose_size(pts, rs):
    if len(rs) != len(pts):
        raise ValueError("Expected number of rotations to be equal to "
                            "number of positions given, got {} points "
                            "and {} rotations."
                            .format(len(pts), len(rs)))

class LidarInfo():
    def __init__(self, resolution=16, distance=30, std_dev=1., fov=[-np.pi,np.pi]):
        self.resolution = resolution
        self.distance = distance
        self.std_dev = std_dev
        self.fov = fov
    
    def __str__(self):
        return "resolution: {}, distance: {}, std_dev: {}, fov: {}".format( \
            self.resolution, self.distance, self.std_dev, self.fov)

def createLidarSegments(lidar_ps, lidar_rs, lidar_info, distance=None):
    """
    Parameters
    ----------
    lidar_ps : array_like, shape (N,2) or (N,3) 
    lidar_rs : `Rotation` instance
        must contain N instances
    lidar_info : `LidarInfo` instance
    

    Returns
    -------
    segments: array_like, shape (N * lidar_resolution, 2, 2), memory (N, lidar_resolution, 2, 2)
    """
    _assert_pose_size(lidar_ps, lidar_rs)
    
    start_pts = np.ndarray([lidar_ps.shape[0], lidar_info.resolution, 2])
    for i in range(lidar_ps.shape[0]):
        start_pts[i,:,:] = lidar_ps[i]
    
    start_pts = start_pts.reshape([lidar_ps.shape[0]* lidar_info.resolution, 2])

    base_angles = np.outer(lidar_rs.as_rotvec()[:,2], np.ones(lidar_info.resolution))
    scan_angles = np.outer(np.ones(len(lidar_rs)), np.linspace(lidar_info.fov[0], lidar_info.fov[1], lidar_info.resolution))

    angles = (base_angles + scan_angles).ravel()

    rotvecs = np.zeros([len(angles),3])
    rotvecs[:,2] = angles
    distance_vecs = np.zeros([len(angles),3])
    if distance is None:
        distance_vecs[:,0] = lidar_info.distance
    else:
        distance_vecs[:,0] = distance

    end_pts = start_pts + R.from_rotvec(rotvecs).apply(distance_vecs)[:,:2]
    return np.ascontiguousarray(np.array([start_pts, end_pts]).transpose(1,0,2))

def Lidar2DMeasure(lidar_ps, lidar_rs, lidar_info, poly_map):
    """
    Returns
    -------
    dist: cl_array, shape (lidar_num * lidar_resolution), can be reshaped to (lidar_num, lidar_resolution)
    """
    if lidar_ps.shape == (2,) or lidar_ps.shape == (3,):
        lidar_ps = np.array([lidar_ps])
    if len(lidar_rs.as_quat().shape) == 1:
        lidar_rs = R.from_quat([lidar_rs.as_quat()])
    rkt = CLRocket.ins()

    
    lidar_segments = createLidarSegments(lidar_ps, lidar_rs, lidar_info)
    beam_num = len(lidar_segments)
    beam_seg_dev = cl_array.to_device(rkt.queue, lidar_segments)
    beam_eqt_dev = cl_array.zeros(rkt.queue, (beam_num * 4, ) , np.float64)

    logging.info("lineEquationFromPointPairs")
    rkt.prg['particle_filter'].lineEquationFromPointPairs( \
        rkt.queue, (beam_num, ), None, \
        beam_seg_dev.data, beam_eqt_dev.data)

    idx_pair_dev = cl_array.zeros(rkt.queue, (beam_num * poly_map.seg_num // 2, 2), np.uint32)
    buff_len_dev = cl_array.zeros(rkt.queue, (1, ), np.uint32)

    logging.info("findIntersected, beam_num: {}, poly_map.seg_num: {}".format(beam_num, poly_map.seg_num))
    rkt.prg['particle_filter'].findIntersected( \
        rkt.queue, (beam_num, poly_map.seg_num), None, \
        beam_seg_dev.data, poly_map.seg_dev.data, idx_pair_dev.data, buff_len_dev.data)
    intersected_num = buff_len_dev.get()[0]

    # -----------------------
    logging.info("calculateDisance")
    init_distances = np.ones(beam_num) * (lidar_info.distance + 1.)
    distance_dev = cl_array.to_device(rkt.queue, init_distances)
    rkt.prg['particle_filter'].calculateDisance( \
        rkt.queue, (intersected_num, ), None, \
        beam_seg_dev.data, idx_pair_dev.data, beam_eqt_dev.data, poly_map.eqt_dev.data, distance_dev.data)
    
    return distance_dev

def lidar2DLikelihood(dev_measure_real, dev_measure_samples, lidar_info, sample_num):
    """
    ---------
    output:
    self.log_likelihood_dev: 1D array, shape (sample_num, )
    """
    rkt = CLRocket.ins()
    beam_num = sample_num * lidar_info.resolution
    beam_likelihood_dev = cl_array.zeros(rkt.queue, (beam_num, ), np.float64)
    log_likelihood_dev = cl_array.zeros(rkt.queue, (sample_num, ), np.float64)

    logging.info("batch likelihood")
    rkt.prg['particle_filter'].BatchLidarLogLikeliHood( \
        rkt.queue, (lidar_info.resolution,), None, \
        dev_measure_real.data, dev_measure_samples.data, \
        np.float64(lidar_info.std_dev), np.float64(lidar_info.distance), np.uint32(sample_num), \
        beam_likelihood_dev.data)
    
    logging.info("reduce sum")
    rkt.prg['particle_filter'].reduceSumRowF64( \
        rkt.queue, (rkt.max_work_group, sample_num), (rkt.max_work_group, 1), \
        beam_likelihood_dev.data, log_likelihood_dev.data, np.uint32(lidar_info.resolution))

    return log_likelihood_dev

if __name__ == "__main__":
    map_lib = os.path.dirname(os.path.abspath(__file__)) + '/map_lib.yaml'
    poly_map = PolygonMap(map_lib, 'basic')

    lidar_reso = 128
    lidar_pts = np.array([[30,30]])
    lidar_rs = R.from_rotvec([[0,0,np.pi]])

    lidar_info = LidarInfo(resolution=lidar_reso, distance=50, fov=[-np.pi/3, np.pi/3])
    lidar_dist = Lidar2DMeasure(lidar_pts, lidar_rs, lidar_info, poly_map)
    beam_viz = createLidarSegments(lidar_pts, lidar_rs, lidar_info, lidar_dist.get())
    
    for seg in poly_map.segments:
        plt.plot(seg[:, 0], seg[:, 1], 'k')
    for seg in beam_viz:
        plt.plot(seg[:, 0], seg[:, 1], 'r', lw=0.3)
    plt.axis('equal')
    plt.show()