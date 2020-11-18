#!/usr/bin/env python
import roslib
roslib.load_manifest('learning_tf')
import rospy

import tf 
#import turtlesim.msg
from tf_broadcast.msg import Vendor

def handle_turtle_pose(msg):
     br = tf.TransformBroadcaster()
     br.sendTransform((msg.pose.x, msg.pose.y, 0),
                      tf.transformations.quaternion_from_euler(0, 0, msg.pose.theta),
                      rospy.Time.now(),
                      msg.vendor_name,
                      "world")

if __name__ == '__main__':
     rospy.init_node('turtle_tf_broadcaster')
     turtlename = 'vendor'
     rospy.Subscriber('/%s/pose' % turtlename,
                      Vendor, # Change this to a message type that includes vendor name and pose
                      handle_turtle_pose)
     rospy.spin()
