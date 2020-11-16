import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)
    t = np.zeros(len(path))
    for i in range(len(path)-1):
        t[i+1] = t[i] + np.linalg.norm(path[i+1,:] - path[i,:]) / V_des
    t_max = t[-1]
    t_smoothed = np.arange(0.0, t_max, dt)
    traj_smoothed = np.zeros((len(t_smoothed), 7))
    sply = scipy.interpolate.splrep(x=t, y=path[:,1], s=alpha)
    splx = scipy.interpolate.splrep(x=t, y=path[:,0], s=alpha)
    
    traj_smoothed[:, 0] = scipy.interpolate.splev(t_smoothed, splx, der=0) #x
    traj_smoothed[:, 1] = scipy.interpolate.splev(t_smoothed, sply, der=0) #y
    traj_smoothed[:, 3] = scipy.interpolate.splev(t_smoothed, splx, der=1) #xd
    traj_smoothed[:, 4] = scipy.interpolate.splev(t_smoothed, sply, der=1) #yd
    traj_smoothed[:, 2] = np.arctan(traj_smoothed[:,4]/ traj_smoothed[:,3]) #theta
    traj_smoothed[:, 5] = scipy.interpolate.splev(t_smoothed, splx, der=2) #xdd
    traj_smoothed[:, 6] = scipy.interpolate.splev(t_smoothed, sply, der=2) #ydd
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
