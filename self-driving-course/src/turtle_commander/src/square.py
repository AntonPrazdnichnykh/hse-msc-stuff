#! /usr/bin/python3.8

import rospy
from geometry_msgs.msg import Twist

rospy.init_node("turtle_square")

pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
msg = Twist()
msg.linear.x = 1.0
msg.angular.z = 1.0

r = rospy.Rate(1)

while not rospy.is_shutdown():
	r.sleep()
	pub.publish(msg)

