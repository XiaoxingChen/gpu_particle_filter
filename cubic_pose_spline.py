from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from scipy.interpolate import CubicSpline
import numpy as np

def _assert_point_size(times, pts):
    if len(times) != len(pts):
        raise ValueError("Expected number of rotations to be equal to "
                            "number of timestamps given, got {} points "
                            "and {} timestamps."
                            .format(len(pts), len(times)))

def _assert_rotation_size(times, rotations):
    if len(times) != len(rotations):
        raise ValueError("Expected number of rotations to be equal to "
                            "number of timestamps given, got {} rotations "
                            "and {} timestamps."
                            .format(len(rotations), len(times)))

class CubicPoseSpline(object):
    """Interpolate C2 continuous pose.

    Parameters
    ----------
    times : array_like, shape (N,)
        Times of the known rotations. At least 2 times must be specified.
    positions : 
        Positions to perform the interpolation between. Must contain N
        positions.
    rotations : `Rotation` instance
        Rotations to perform the interpolation between. Must contain N
        rotations.

    Methods
    -------
    lin_p
    lin_v
    lin_a
    ang_p
    ang_v
    ang_a

    Examples
    --------
    >>> from cubic_pose_spline import CubicPoseSpline
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation as R

    Define the sequence of times, positions and rotations

    >>> ts = np.array([1,2,3,4,6,8,9,10])
    >>> ps = np.array(np.zeros([len(ts), 3]))
    >>> rs = R.random(len(ts))
    >>> cusp = CubicPoseSpline(ts, ps)

    Interpolate all dynamic values:

    >>> t_sample = np.array(ts[0], ts[-1], 50)
    >>> position = cusp.lin_p(t_sample)
    >>> velocity = cusp.lin_v(t_sample)
    >>> acceleration = cusp.lin_a(t_sample)
    >>> orientation = cusp.ang_p(t_sample)
    >>> angular_rate = cusp.ang_v(t_sample)
    >>> angular_acceleration = cusp.ang_a(t_sample)
    """
    def __init__(self, ts, pts, rs=None):
        _assert_point_size(ts, pts)
        if rs is None:
            rs = R.identity(ts.shape[0])
        else:
            _assert_rotation_size(ts, rs)

        self._rot_spline = RotationSpline(ts, rs)
        self._trans_spline = CubicSpline(ts, pts)
        self._point_dim = pts.shape[1]

    def _rotation(self, times, position=None, orientation=None, order=0):
        """
        Compute rotation at time t.
        Parameters
        ----------
        times : array_like, shape (N,)
            Times of the known target poses. At least 2 times must be specified.
        order : {0, 1, 2}, optional
            Order of differentiation.
        position : array_like, shape (2,) or (3,)
            Target position in body frame.
        orientation : `Rotation` instance or None
            Target orientation in body frame.

        Returns
        -------
        Interpolated orientation, angular rate or angular acceleration at target time.
        """

        value_origin = self._rot_spline(times, order)
        if orientation is None:
            return value_origin

        if 0 == order:
            value_local = value_origin * orientation
        elif 1 == order:
            value_local = value_origin # angular rate is same in a rigid body
        elif 2 == order:
            value_local = value_origin
        else:
            assert False

        return value_local

    def _translation(self, times, position=None, orientation=None, order=0):
        """
        Compute translation at time t.
        Parameters
        ----------
        times : array_like, shape (N,)
            Times of the known target poses. At least 2 times must be specified.
        order : {0, 1, 2}, optional
            Order of differentiation.
        position : array_like, shape (2,) or (3,)
            Target position in body frame.
        orientation : `Rotation` instance or None
            Target orientation in body frame.

        Returns
        -------
        Interpolated position, velocity or acceleration at target time.
        """

        value_origin = self._trans_spline(times, order)
        if position is None and orientation is None:
            return value_origin

        if position is None:
            position = np.zeros((self._point_dim,))

        if orientation is None:
            orientation = R.identity(1)

        r_wb = self._rot_spline(times, 0) # orientation, body to world
        omega_w = self._rot_spline(times, 1) # angular_rate in world frame

        if 0 == order:
            value_local = value_origin + r_wb.apply(position)
            
        elif 1 == order:
            value_local = value_origin + np.cross(omega_w, r_wb.apply(position))

        elif 2 == order:
            angular_acc_w = self._rot_spline(times, 2)

            # angular acceleration in world frame
            value_local = value_origin 

            # central pedal acceleration : a = w x (w x r)
            value_local += np.cross(omega_w, np.cross(omega_w, r_wb.apply(position)))

            # Euler acceleration : w' x r
            value_local += np.cross(angular_acc_w, r_wb.apply(position))
        else:
            assert False

        return value_local

    def lin_p(self, times, pos=None):
        return self._translation(times, position=pos, orientation=None, order=0)

    def lin_v(self, times, pos=None):
        return self._translation(times, position=pos, orientation=None, order=1)

    def lin_a(self, times, pos=None):
        return self._translation(times, position=pos, orientation=None, order=2)

    def ang_p(self, times, ori=None):
        return self._rotation(times, position=None, orientation=ori, order=0)

    def ang_v(self, times):
        return self._rotation(times, position=None, orientation=None, order=1)

    def ang_a(self, times):
        return self._rotation(times, position=None, orientation=None, order=2)

class SplineIMU():
    """Sample IMU data from C2 continuous pose spline.

    Parameters
    ----------
    pose_spline : C2 continue pose spline
    local_p : 
        Position of IMU sensor with respect to the center of pose spline.
    local_r : `Rotation` instance
        Orientation of IMU sensor with respect to the center of pose spline.
    gravity : gravity in global coordinate
        
    Methods
    -------
    data

    Examples
    --------
    >>> from cubic_pose_spline import CubicPoseSpline
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation as R

    Define the sequence of times, positions and rotations

    >>> ts = np.array([1,2,3,4,6,8,9,10])
    >>> ps = np.array(np.zeros([len(ts), 3]))
    >>> rs = R.random(len(ts))
    >>> cusp = CubicPoseSpline(ts, ps)

    Define Spline IMU

    >>> imu = SplineIMU(cusp, np.array([0,0,0.5]), R.from_rotvec([0, np.pi, 0]))
    >>> t_sample = np.array(ts[0], ts[-1], 50)
    >>> imu_data = imu.data(t_sample)
    >>> imu_acceleration = imu_data.acc
    >>> imu_angular_rate = imu_data.gyr
    
    """
    class ImuData():
        def __init__(self, acc, gyr):
            self.acc = acc
            self.gyr = gyr

    def __init__(self, pose_spline, local_p=np.zeros(3), local_r=R.identity(), gravity=np.array([0,0,9.81])):
        self.p = local_p
        self.r = local_r
        self.spline = pose_spline
        self.gravity_global = gravity
    
    def data(self, times):
        # rotation: IMU to world
        r_wi = self.spline.ang_p(times, ori=self.r)

        world_acc = self.spline.lin_a(times, pos=self.p) + self.gravity_global
        world_gyr = self.spline.ang_v(times)

        imu_acc = r_wi.inv().apply(world_acc)
        imu_gyr = r_wi.inv().apply(world_gyr)
        return SplineIMU.ImuData(imu_acc, imu_gyr)
    