import unittest
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from lidar_2d import *
from polygon_map import *
import matplotlib.pyplot as plt
from cl_rocket import *
from scipy.stats import norm
import logging

class TestLidar2D(unittest.TestCase):
    def test_lidar_2d_measure(self):
        return
        # poly_map = PolygonMap('circle_corridor')
        poly_map = PolygonMap('corridor')
        lidar_info = LidarInfo(resolution=2)
        lidar_num = 1
        lidar_ps = np.ones([lidar_num,2])
        lidar_rs = R.random(lidar_num)
        for i in range(1000):
            print("i: {}".format(i))
            result = Lidar2DMeasure(lidar_ps, lidar_rs, lidar_info, poly_map)

    def test_find_intersected(self):
        poly_map = PolygonMap('corridor')
        lidar_info = LidarInfo(resolution=256)
        lidar_num = 256
        lidar_ps = np.ones([lidar_num,2])
        lidar_rs = R.random(lidar_num)

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

        for i in range(10000):
            buff_len_dev = cl_array.zeros(rkt.queue, (1, ), np.uint32)
            logging.info("findIntersected, beam_num: {}, poly_map.seg_num: {}".format(beam_num, poly_map.seg_num))
            rkt.prg['particle_filter'].findIntersected( \
                rkt.queue, (beam_num, poly_map.seg_num), None, \
                beam_seg_dev.data, poly_map.seg_dev.data, idx_pair_dev.data, buff_len_dev.data)
            intersected_num = buff_len_dev.get()[0]
            print("i: {}".format(i))

if __name__ == '__main__':
    logging.basicConfig(filename='gpu_pf_test.log', 
        format='[%(asctime)s] - %(message)s', level=logging.INFO, filemode='w')
    unittest.main()