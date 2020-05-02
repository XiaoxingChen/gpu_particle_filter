#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void findIntersected(
    __global double4* src_segments,
    __global double4* dst_segments,
    __global int2* out_pairs,
    __global int* out_len)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int stride = get_global_size(1);

    double4 s0d0 = (double4)(dst_segments[j].xy - src_segments[i].xy, 0, 0);
    double4 s0d1 = (double4)(dst_segments[j].zw - src_segments[i].xy, 0, 0);
    double4 s0s1 = (double4)(src_segments[i].zw - src_segments[i].xy, 0, 0);
    double4 d0s1 = (double4)(src_segments[i].zw - dst_segments[j].xy, 0, 0);
    double4 d0d1 = (double4)(dst_segments[j].zw - dst_segments[j].xy, 0, 0);

    double4 d0s0 = -s0d0;

    uchar cond_0 = dot(cross(s0d0, s0s1), cross(s0s1, s0d1)) > 0;
    uchar cond_1 = dot(cross(d0s0, d0d1), cross(d0d1, d0s1)) > 0;

    if (! (cond_0 && cond_1)) return;

    int out_idx = atomic_inc(out_len);
    out_pairs[out_idx] = (int2)(i, j);
}

void atomic_min_global(volatile __global double *source, const double operand) {
    union {
        ulong intVal;
        double floatVal;
    } newVal;
    union {
        ulong intVal;
        double floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = min(prevVal.floatVal, operand);
    } while (atom_cmpxchg((volatile global ulong *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void calculateDisance(
    __global double4* src_segments,
    __global int2* idx_pairs,
    __global double4* eqt_src,
    __global double4* eqt_dst,
    __global double* min_dist_buff)
{
    int out_idx = get_global_id(0);
    int src_idx = idx_pairs[out_idx].x;
    int dst_idx = idx_pairs[out_idx].y;

    double4 mat_a = (double4)(eqt_src[src_idx].xy, eqt_dst[dst_idx].xy);
    double2 vec_b = (double2)(eqt_src[src_idx].z , eqt_dst[dst_idx].z);

    double det_a = mat_a.x * mat_a.w - mat_a.y * mat_a.z;
    double4 inv_a = (double4)(mat_a.w, -mat_a.y, -mat_a.z, mat_a.x) / det_a;
    double2 intersect_pt = (double2) (dot(inv_a.xy, vec_b), dot(inv_a.zw, vec_b));
    double dist = length(intersect_pt - src_segments[src_idx].xy);

    atomic_min_global(&min_dist_buff[src_idx], dist);
}

__kernel void lineEquationFromPointPairs(__global double4* pt_pairs , __global double4* equation)
{
    int gid = get_global_id(0);
    double4 pt_pair = pt_pairs[gid];
    double2 tan_v = pt_pair.zw - pt_pair.xy;
    //norm = (-y, x)
    double4 norm_v = normalize((double4)(-tan_v.y, tan_v.x, 0, 0));
    double d = dot(norm_v.xy, pt_pair.xy);
    equation[gid] = (double4)(norm_v.xy, d, 0);
}

double normCDF(double mean, double std_dev, double x)
{
    return 0.5 * (1 + erf((x - mean) * M_SQRT1_2 /std_dev));
}

double gauss(double mean, double std_dev, double x)
{
    double normed_x = (x - mean) / std_dev;
    return (M_SQRT1_2 * M_2_SQRTPI)/(2 * std_dev) * exp(-.5 * normed_x * normed_x);
}

double LidarLogLikeliHood(double mean, double std_dev, double x, double up_limit)
{
    double likelihood = 0.;
    double low_limit = -1.;
    if(x > up_limit)
    {
        likelihood = (1 - normCDF(mean, std_dev, up_limit));
    }
    else
    {
        likelihood = gauss(mean, std_dev, x);
    }
    return max(log(likelihood), low_limit);
}

/** calculate LiDAR particle likelihood with real measurement
 *
 * @param[in]mean         1D array: shape (N, )   N is the resolution of LiDAR.
 * @param[in]x            2D array: shape (M, N). M is the number of particle.
 * @param[in]std_dev      double: standard deviation of LiDAR sampling.
 * @param[in]up_limit     double: max range of LiDAR sensor.
 * @param[in]particle_num uint: number of particles, particle_num == M.
 * @param[out]likelihood   2D array: shape (M, N). likelihood of each LiDAR beam of each particle.
 * 
 * @note : the function should be called with global index (N, ), so that `lidar_reso == get_global_size(0)`.
 *
 * @return void.
 */
__kernel void BatchLidarLogLikeliHood(
    __global double* mean, 
    __global double* x, 
    double std_dev, 
    double up_limit,
    uint particle_num,
    __global double* likelihood)
{
    int lidar_reso = get_global_size(0);
    int idx_beam = get_global_id(0);

    for(int i = 0; i < particle_num; i++)
    {
        likelihood[lidar_reso * i + idx_beam] = //log(0.3989422804014327);
            LidarLogLikeliHood(mean[idx_beam], std_dev, x[lidar_reso * i + idx_beam], up_limit);
        // printf("x: %f\n", x[lidar_reso * i + idx_beam]);
    }
}

__kernel void testNormCDF(__global double* mean, __global double* std_dev, __global double* x, __global double* out)
{
    int i = get_global_id(0);
    out[i] = normCDF(mean[i], std_dev[i], x[i]);
}

/** cumulative product along row vector
 *
 * @param[in]g            2D array: shape (H, W)  
 * @param[in]out          1D array: shape (H,  )
 * @param[in]width          uint: length of row vector, `W = width`.
 * 
 * @note : the function should be called with global index (MAX_WORK_GROUP_NUM, H), local index (MAX_WORK_GROUP_NUM, 1).
 *
 * @return void.
 */
__kernel void reduceSumRowF64(__global double* g, __global double* out, int width)
{
    uint lsize = get_local_size(0);
    uint lid   = get_local_id(0);
    uint idx_h = get_global_id(1);
    uint remainder = width % lsize;
    uint repeat = width / lsize;

    __global double* g_local = &g[idx_h * width];


    __local double local_results[1024];
    local_results[lid] = 0.;
    barrier(CLK_LOCAL_MEM_FENCE );

    for(int i = 0; i < repeat; i++)
    {
        local_results[lid] += g_local[i * lsize + lid + remainder];
    }
    if(lid < remainder)
    {
        local_results[lid] += g_local[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE );

    // Perform parallel reduction
    for(unsigned int i = lsize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_results[lid] += local_results[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        out[idx_h] = local_results[0];
        // printf("idx_h: %d, local_results[0]: %f\n", idx_h, local_results[0]);
    }
    return;
}

#if 0
double4 outer(double2* a,  double2* b)
{
    double xx = q2d[idx].x * q2d[idx].x;
    double xy = q2d[idx].x * q2d[idx].y;
    double yy = q2d[idx].y * q2d[idx].y;
    return (double4)(xx, xy, xy, yy);
}
/** calculate weighted mean of so2
 *
 * @param[in]angle          1D array: shape (N, )  
 * @param[in]weight         1D array: shape (N, )
 * @param[in]len            uint: length of weight vector, `N = len`.
 * @param[xx]q2d            double2 array: shape (N, ). 2D quaternion for storage
 * @param[out]out           1D array: shape (4, ). 
 *                          Output 2x2 matrix. the mean 2d quaternion is the last eigenvector of the matrix.
 * 
 * @note : the function should be called with global index (MAX_WORK_GROUP_NUM, ), local index (MAX_WORK_GROUP_NUM, ).
 *
 * @return void.
 */
__kernel void weightedMeanSO2(
    __global double* angle, 
    __global double* weight, 
    int len, 
    __local double2* q2d,
    __global double* out)
{
    uint lsize = get_local_size(0);
    uint lid = get_local_id(0);
    uint remainder = len % lsize;
    uint repeat = len / lsize;

    // construct q2d
    for(int i = 0; i < repeat; i++)
    {
        int idx = remainder + i * lsize + lid;
        q2d[idx] = (double2)(cos(angle[idx]/2), sin(angle[idx]/2));
    }
    if(lid < remainder)
    {
        q2d[lid] = (double2)(cos(angle[lid]/2), sin(angle[lid]/2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //calculate np.dot(q.T, q)
    double4 qmat = (double4)(0,0,0,0);
    for(int i = 0; i < repeat; i++)
    {
        int idx = remainder + i * lsize + lid;
        qmat += outer(&q2d[idx], &q2d[idx]) * weight[idx];
    }
    if(lid < remainder)
    {
        qmat += outer(&q2d[lid], &q2d[lid]) * weight[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce sum
    __local double4 local_results[1024];
    local_results[lid] = qmat;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int i = lsize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_results[lid] += local_results[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        out[0] = local_results[0].x;
        out[1] = local_results[0].y;
        out[2] = local_results[0].z;
        out[3] = local_results[0].w;
    }
}

#endif