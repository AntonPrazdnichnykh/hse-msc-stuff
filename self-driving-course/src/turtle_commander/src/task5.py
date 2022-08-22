#! /usr/bin/python3.8

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import atan, pi


class TurtleChaser:
	def __init__(self):
		rospy.Subscriber('/turtle1/pose', Pose, self.callback1)
		rospy.Subscriber('/turtle2/pose', Pose, self.callback2)
		self.pub2 = rospy.Publisher('/turtle2/cmd_vel', Twist, queue_size=10)
		self.x2 = 0
		self.y2 = 0
		self.theta2 = 0

	def callback1(self, msg):
		x, y = msg.x, msg.y
		msg_vel = Twist()
		delta_x = x - self.x2
		delta_y = y - self.y2
		msg_vel.angular.z = -self.theta2 + atan(delta_y / (delta_x + 1e-8))
		self.pub2.publish(msg_vel)
		msg_vel.angular.z = 0.0
		msg_vel.linear.x = delta_x
		msg_vel.linear.y = delta_y
		self.pub2.publish(msg_vel)
		

	def callback2(self, msg):
		self.x2 = msg.x
		self.y2 = msg.y
		self.theta2 = msg.theta
		# rospy.loginfo(f"x2: {self.x2}, y2: {self.y2}, theta2: {self.theta2}")
		

rospy.init_node("turtle_chaser")
TurtleChaser()
rospy.spin()
