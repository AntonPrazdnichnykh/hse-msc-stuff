#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np


class LaserMap:
    def __init__(self, eps = 0.01):
        rospy.Subscriber('/base_scan', LaserScan, self.callback)
        self._eps = eps
        self.marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        # self.map_publisher = rospy.Publisher("/map_topic", OccupancyGrid, queue_size=10)
        self.rate = rospy.Rate(1)

    def callback(self, msg):
        r = np.array(msg.ranges)
        phi = msg.angle_min + msg.angle_increment * np.arange(len(r))
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        mask = (np.diff(x) ** 2 + np.diff(y) ** 2) < self._eps
        mask = np.concatenate(([True], mask))
        
        x, y = x[mask], y[mask]
        
        marker = Marker()
        
        marker.header.frame_id = "base_laser_link"

        marker.type = 8
        marker.id = 0
        marker.action = 0
        
        # Set the scale of the marker
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

	# Set the color
        marker.color.r = 1.
        marker.color.g = 1.
        marker.color.b = 1.
        marker.color.a = 0.7

        # Set the pose of the marker
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        
        marker.points = [Point(x_, y_, 0.) for x_, y_ in zip(x, y)]

        self.rate.sleep()

        self.marker_publisher.publish(marker)
        
        

rospy.init_node('laser_scan_publisher')
LaserMap(eps=0.05)
rospy.spin()

