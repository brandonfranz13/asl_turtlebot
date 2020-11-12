import rospy
import numpy as np
import math 
from utils import wrapToPi
from std_msgs.msg import Float64

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        
	x_shift = x-self.x_g
        y_shift = y-self.y_g
        th

        # Find rho by taking the l-2 norm along the rows
        rho = np.apply_along_axis(np.linalg.norm, 1, np.c_[x_shift, y_shift])
        alpha = wrapToPi(np.arctan2(y_shift, x_shift)-th+np.pi)
        delta = wrapToPi(alpha+th)

        # Calculated control inputs (we divide by np.pi because of sinc definition)
        V = self.k1*rho*np.cos(alpha)
        om = self.k2*alpha+self.k1*np.sinc(alpha/np.pi)*np.cos(alpha)*(alpha+self.k3*delta)
        
	alphaPub = rospy.Publisher('/controller/alpha', Float64, queue_size=10)
        deltaPub = rospy.Publisher('/controller/delta', Float64, queue_size=10)
        rhoPub = rospy.Publisher('/controller/rho', Float64, queue_size=10)


	alphaMsg = Float64()
	deltaMsg = Float64()
	rhoMsg = Float64() 

	alphaMsg.data = alpha
	deltaMsg.data = delta
	rhoMsg.data = rho

	alphaPub.publish(alphaMsg)
	deltaPub.publish(deltaMsg)
	rhoPub.publish(rhoMsg)

  	########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
