#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from asl_turtlebot.msg import DetectedObject
from tf_broadcast.msg import Vendor
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0 #agent is awaiting instructions
    ALIGN = 1 #NAV part 1: heading controller
    TRACK = 2 #NAV part 2: trajectory tracker
    PARK = 3 #NAV part 3: pose controller
    STOP = 4 #stopped in front of a stop sign
    CROSS = 5 #moving while ignoring stop sign
    MEOW = 6 #broadcasting message upon detecting a cat
    PICKUP = 7 #pausing at a goal location
    RTB = 8 #set goal to initial position to prepare for deliveries
    START_DELIVERY = 9 #change mode from EXPLORATION to DELIVERY

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 #initial state is set as point of origin for x, y and theta
        
        self.home = (0,0,0)

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]

        # Robot limits
        self.v_max = rospy.get_param("~v_max", 0.2)    # maximum velocity
        self.om_max = rospy.get_param("~om_max", 0.4)   # maximum angular velocity

        self.v_des = 0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.01
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0.4, 0.8, 0.8, self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        # indicators for state machine additions
        self.delivery_mode = False #0 is EXPLORATION, 1 is DELIVERY
        self.fully_explored = rospy.get_param("~fully_explored", False) #whether or not space is judged as fully explored
        self.has_meowed = False #whether or not we have meowed/cheered at a currrently visible cat/beer
        self.detectedStopSign = False
        self.detectedCat = False
        self.stop_time = rospy.get_param("~stop_time", 3.) # Time to stop at a stop sign
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.8) # Proximity from a stop sign to obey it
        self.pickup_time = rospy.get_param("~pickup_time", 4.) # Time taken to pick food up at a vendor between 3 and 5 seconds

        # Obstacle avoidance
        self.laser_ranges = []
        self.collisionImminent = False
        self.collisionThreshold = 0.008 # was 0.15 previously
        self.obstacle_padding = 0.4
        self.laser_angle_increment = 0.1
        
        # Vendor Catalogue
        self.vendor_catalogue = {}
        
        # list of goals, in order
        self.goal_list = [] #implement as a list of three-element tuples, with the last tuple being all zeroes (original position)

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        #Delivery Request Subscriber
        rospy.Subscriber('/delivery_request', String, self.delivery_request_callback)        
        
        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # Beer detector
        rospy.Subscriber('/detector/beer', DetectedObject, self.cat_detected_callback) #detecting beer, not cat

        # Subscribe to vendors
        rospy.Subscriber('/detector/giraffe', DetectedObject, self.vendor_callback)
        rospy.Subscriber('/detector/horse', DetectedObject, self.vendor_callback)
        rospy.Subscriber('/detector/bear', DetectedObject, self.vendor_callback)
        rospy.Subscriber('/detector/zebra', DetectedObject, self.vendor_callback)
        rospy.Subscriber('/detector/cow', DetectedObject, self.vendor_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.vendor_callback)

        # Publisher for "meow" message
        self.messages = rospy.Publisher('/mensaje', String, queue_size=10)
        mensaje = String()
        mensaje.data = "Saluti!" #message to broadcast in "meow" state

        print "finished init"

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        self.spline_alpha = config["alpha"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan() # new map, need to replan

    def shutdown_callback(self):
        """ publishes zero velocities upon rospy shutdown """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)
        
    def laser_callback(self, msg):
        """ provides range data for obstacle avoidance """
        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment
        self.collisionImminent = np.any([range < self.collisionThreshold for range in self.laser_ranges])
        
    def delivery_request_callback(self, msg):
        """
        Callback for the delivery request from request_publisher.py. 
        Message format is a string of comma-separated items to pickup and deliver
        """
        def isWaiting():
            return self.delivery_request == None
        
        if isWaiting():
            self.delivery_request = [request.strip() for request in msg.data.split(',')]
            self.goal_list = [self.vendor_catalogue[vendor] for vendor in self.delivery_request, self.home]
            self.goal_list.append(self.home)
            print("Order Received. Out for delivery!")
            self.switch_mode(Mode.PICKUP)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def aligned(self):
        """returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """        
        return (abs(wrapToPi(self.theta - wrapToPi(self.th_init))) < self.theta_start_thresh)
    
    def aligned_to_object(self, object_theta):
        """returns whether robot is aligned to an object
        """      
        print("Aligned to object data")
        print(self.theta)
        print(object_theta)
        print(abs(wrapToPi(self.theta - object_theta)))
        return (abs(wrapToPi(self.theta - object_theta)) < self.theta_start_thresh)

    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution, self.obstacle_padding)

        rospy.loginfo("Navigator: computing navigation plan")
        success =  problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path


        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK) #switch to pose controller
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        print("A few traj_new thetas")
        print(traj_new[0:20,2])
        print(traj_new[-20:, 2])
        self.heading_controller.load_goal(wrapToPi(self.th_init))

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    ## new functions & functions from supervisor.py
    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def init_cat(self):
        """initiates a message broadcast"""
        self.cat_start = rospy.get_rostime()
        self.mode = Mode.MEOW

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def has_picked_up(self):
        """ checks if pickup maneuver is over """

        return self.mode == Mode.PICKUP and \
               rospy.get_rostime() - self.pickup_start > rospy.Duration.from_sec(self.pickup_time)

    def has_crossed(self):
        """ checks if crossing maneuver is over (stop sign no longer visible)"""

        return self.mode == Mode.CROSS and not self.hasDetectedStopSign()

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in track or park mode, stop
        if self.mode == Mode.TRACK or self.mode == Mode.PARK:
            if dist > 0 and dist < self.stop_min_dist:
                self.init_stop_sign()

    def cat_detected_callback(self, msg):
        """ callback for when the detector has found a cat (or beer). A distance
        of 0 can mean the item was not detected """

        # distance of the cat
        dist = msg.distance

        # if cat detected and in track or park mode, respond
        if self.mode == Mode.TRACK or self.mode == Mode.PARK and dist > 0:
            self.init_cat()

    def vendor_callback(self, msg):
        """ callback for the list of vendors we build 
        object_msg.id = cl
                object_msg.name = self.object_labels[cl]
                object_msg.confidence = sc
                object_msg.distance = dist
                object_msg.thetaleft = thetaleft
                object_msg.thetaright = thetaright
                object_msg.corners = [ymin,xmin,ymax,xmax]
        """
        print("Condition to publish vendor")
        print(self.vendor_catalogue.has_key(msg.name))
        print("I do what I want")
        
        if not self.vendor_catalogue.has_key(msg.name): # make sure we don't change vendor location
            avgTheta = (msg.thetaleft + msg.thetaright) / 2.
            vendor_x = msg.distance * np.cos(avgTheta)
            vendor_y = msg.distance * np.sin(avgTheta)
            vendor_theta = avgTheta
            
            self.vendor_pub = rospy.Publisher('/vendor/pose', Vendor, queue_size=10)
            
            vendor_msg = Vendor()
            vendor_msg.vendor_name = msg.name
            vendor_msg.pose.x = vendor_x
            vendor_msg.pose.y = vendor_y
            vendor_msg.pose.theta = vendor_theta
            
            self.vendor_pub.publish()
            
            # Convert to world coordinates
            (translation,rotation) = self.trans_listener.lookupTransform('base_camera', 'world', rospy.Time(0))
            vendor_x = translation[0]
            vendor_y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            vendor_theta = euler[2]
            
            self.vendor_catalogue[msg.name] = (vendor_x, vendor_y, vendor_theta)
            
            print("Vendor Catalogue")
            print(msg.name)
            print(self.vendor_catalogue[msg.name])
    
    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def at_origin(self):
        """
        returns whether the robot is close enough in position to the original
        position to return to idle state
        """
        return (linalg.norm(np.array([self.x-0.0, self.y-0.0])) < self.near_thresh and abs(wrapToPi(self.theta - 0.0)) < self.at_thresh_theta)

    def goal_origin(self):
        """
        returns whether the current goal of the robot is the original position
        """
        return (linalg.norm(np.array([self.x_g-0.0, self.y_g-0.0])) < self.near_thresh and abs(wrapToPi(self.theta_g - 0.0)) < self.at_thresh_theta)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.nav_vel_pub.publish(vel_g_msg)
        
    def backup(self, decay=0.8, time_to_backup = 0.25):
        """ Put robot in reverse """
        start = rospy.get_time()
        velocity = -self.v_max
        while (rospy.get_time() - start) < time_to_backup:
            cmd_vel = Twist()
            velocity = decay * velocity
            cmd_vel.linear.x = velocity
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)

    def pass_sign(self):
        """ move, ignoring stop sign """
        if self.near_goal():
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)
        
    def update_state(self):
        (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        self.x = translation[0]
        self.y = translation[1]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta = euler[2]
        
    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                self.update_state()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            ################# IDLE ####################
            if self.mode == Mode.IDLE: #awaiting instructions
                #pass
                if self.fully_explored and not self.delivery_mode: #in exploration mode, we are notified the environment is fully explored
                    self.mode = Mode.RTB #return to initial position for transition to delivery mode
                    
            ################# ALIGN ####################
            elif self.mode == Mode.ALIGN: #rotating to face the direction indicated by the start of the path
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
                    
            ################# TRACK ####################
            elif self.mode == Mode.TRACK: #use the tracking controller to follow the planned path
##EXTENSION     # Collision Avoidance
                if self.collisionImminent: # backs up until outside collision threshold
                    rospy.loginfo("Collision Imminent: Backing up and replanning")
                    # while self.collisionImminent:
                        # self.backup()
		    print('just for kicks')
		    print(np.any([range < self.collisionThreshold for range in self.laser_ranges]))
                    collision_indices = np.nonzero(np.array(list(self.laser_ranges)) < self.collisionThreshold)
		    print(self.laser_ranges)
		    print('middle step')
		    print(collision_indices)
		    #collision_indices = collision_indices.flatten() 
		    print('collision indices:')
		    print(collision_indices)
		    collision_thetas = self.laser_angle_increment * collision_indices[0]
		    print('collision_thetas')
		    print(collision_thetas)
		    print('WOWZA')
		    print(np.asarray(self.laser_ranges)[collision_indices])
		    
		    weighted_thetas = (1/sum(np.min(np.asarray(self.laser_ranges)[collision_indices])/np.asarray(self.laser_ranges)[collision_indices]))*collision_thetas*(np.min(np.asarray(self.laser_ranges)[collision_indices]))/np.asarray(self.laser_ranges)[collision_indices]   
		    collision_object_theta = sum(weighted_thetas)+np.pi
                    self.heading_controller.load_goal(collision_object_theta)
                    print("YOU SPIN ME RIGHT ROUND")
                    while not self.aligned_to_object(collision_object_theta):
                        self.update_state()
                        V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, 1) #t=1, time not used
                        cmd_vel = Twist()
                        cmd_vel.linear.x = V
                        cmd_vel.angular.z = om
                        self.nav_vel_pub.publish(cmd_vel)
                    print("BACKING UP")    
                    self.backup(0.9995, 0.5)
                    self.stay_idle()
                    self.replan()
                    self.switch_mode(Mode.ALIGN)
                
                elif self.near_goal(): #near goal
                    self.switch_mode(Mode.PARK) #switch to pose controller for final approach

                ## For cats, beers and stop signs
                elif self.has_meowed and not self.detectedCat: #we are not detecting a cat or beer and have already responded
                    self.has_meowed = False #reset in case a new one is detected

                ## For marking complete exploration
                elif self.fully_explored and not self.delivery_mode: #environment is fully explored but we are not near a goal
                    if not self.goal_origin(): #if initial position is not the current goal
                        self.switch_mode(Mode.RTB)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            
            ################# PARK ####################
            elif self.mode == Mode.PARK: #use pose controller for final approach
                ## for picking up food
                if self.delivery_mode and len(self.goal_list) > 1 and self.at_goal(): #we are in delivery mode and this is not the last goal
                    self.mode = Mode.PICKUP

                elif self.at_goal(): #done maneuvering!
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE) #await further instructions

                # For cats, beers and stop signs """                    
                elif self.has_meowed and not self.detectedCat: #we are not detecting a cat or beer and have already responded
                    self.has_meowed = False #reset in case a new one is detected

                #For marking complete exploration"""
                elif self.fully_explored and not self.delivery_mode: #environment is fully explored but we are not at the goal
                    if not goal_origin(): #if initial position is not the current goal
                        self.switch_mode(Mode.RTB)
                        
##EXTENSION ################# STOP ####################
            elif self.mode == Mode.STOP:
                # At a stop sign
                if not self.has_stopped(): #timer hasn't run out yet
                    self.stay_idle() #don't move yet
                else: #timer has run out
                    self.init_crossing() #start crossing
            
            ################# CROSS ####################   
            elif self.mode == Mode.CROSS:
                # Crossing an intersection
                if not self.has_crossed(): #stop sign is still visible
                        self.pass_sign() #keep moving and ignoring sign
                else:
                    self.mode = Mode.TRACK #resume movement
            
##EXTENSION ################# MEOW ####################         
            elif self.mode == Mode.MEOW:
                self.messages.publish(mensaje) #publish the miao message to a dedicated topic
                self.has_meowed = True #to ensure we do not respond repeatedly t
                self.mode = Mode.TRACK # resume movement (without aligning with starting angle)
            
            ################# PICKUP #################### 
            elif self.mode == Mode.PICKUP:
                if not self.has_picked_up():
                    self.stay_idle()
                else:
                    self.x_g = self.goal_list[0][0] #copy the contents of the first tuple as our next goal
                    self.y_g = self.goal_list[0][1]
                    self.theta_g = self.goal_list[0][2]
                    self.goal_list.pop(0) #remove the first tuple from our list of goals
                    self.mode = Mode.ALIGN #resume movement (from beginning)
            
            ################# RETURN TO BASE (RTB) #################### 
            elif self.mode == Mode.RTB:
                self.x_g = 0.0
                self.y_g = 0.0
                self.theta_g = 0.0 #set next goal position to be the point of orign
                self.mode = Mode.ALIGN #resume movement
            
            ################# START DELIVERY #################### 
            elif self.mode == Mode.START_DELIVERY: #this state transitions to delivery mode
                self.delivery_mode = True #switch to delivery mode
                self.mode = Mode.IDLE
            
            rospy.loginfo(self.mode)
            self.publish_control()
            rate.sleep()

if __name__ == '__main__':
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
