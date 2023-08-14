# -*- coding:utf-8 -*-

# -----------------------------------
# HRI Data Plotter
# Author: souljaboy764
# Date: 2022/07/06
# -----------------------------------

import numpy as np
from read_hh_hr_data import *

import os
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Point
from moveit_msgs.msg import DisplayRobotState

rospy.init_node('buetepage_yumi_visualizer_node')

markerarray_msg = MarkerArray()
lines = []
for i in range(4):
	marker = Marker()
	line_strip = Marker()
	line_strip.ns = marker.ns = "nuitrack_skeleton"
	marker.header.frame_id = line_strip.header.frame_id = 'yumi_base_link'
	marker.id = i
	line_strip.id = i + 4
	line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
	line_strip.frame_locked = marker.frame_locked = False
	line_strip.action = marker.action = Marker.ADD

	marker.type = Marker.SPHERE
	line_strip.type = Marker.LINE_STRIP

	line_strip.color.r = marker.color.g = 1
	line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0
	line_strip.color.a = marker.color.a = 1

	marker.scale.x = marker.scale.y = marker.scale.z = 0.075
	line_strip.scale.x = 0.04

	line_strip.pose.orientation = marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

	line_strip.points = [Point(), Point()]

	markerarray_msg.markers.append(marker)
	lines.append(line_strip)
lines = lines[:-1]
markerarray_msg.markers = markerarray_msg.markers + lines

joint_names = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r', 'yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l']
joint_l = np.deg2rad([-50, -27, 133, 24, 234, 72, 38]).tolist()


trajectory_msg = DisplayRobotState()
trajectory_msg.state.joint_state.header.frame_id = 'yumi_base_link'
trajectory_msg.state.joint_state.name = joint_names

rate = rospy.Rate(100)
robot_pub = rospy.Publisher('display_robot_state', DisplayRobotState, queue_size=10)
human_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

src_dir = './hr'
actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']
idx_list = np.array([joints_dic[joint] for joint in ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])

for action in actions:
	data_h, data_r = read_hri_data(action, src_dir = './hr/')

	seq_len, _, dims = data_h.shape
	data_h = data_h[:, idx_list]#.reshape((-1, len(idx_list)*dims))
	data_h[..., [0,1,2]] = data_h[..., [2,0,1]]
	data_h[..., 1] *= -1			
	segments_file_h = os.path.join(src_dir, 'segmentation', action+'_p1.npy')
	segments_file_r = os.path.join(src_dir, 'segmentation', action+'_r2.npy')
	segments_h = np.load(segments_file_h)
	segments_r = np.load(segments_file_r)

	for i in range(len(segments_h)):
		s_r = segments_r[i]
		s_h = segments_h[i]
		robot_data = data_r[s_r[0]:s_r[1]]
		human_data = data_h[s_h[0]:s_h[1]]
	
		for s in range(s_r[1]-s_r[0]):
			stamp = rospy.Time.now()
			for m in range(4):
				markerarray_msg.markers[m].pose.position.x = 0.7 - human_data[s][m][0]
				markerarray_msg.markers[m].pose.position.y = -0.1 - human_data[s][m][1]
				markerarray_msg.markers[m].pose.position.z = human_data[s][m][2] - 0.7
				if m>0:
					line_idx = 4 + m - 1
					markerarray_msg.markers[line_idx].points[0].x = 0.7 - human_data[s][m-1][0]
					markerarray_msg.markers[line_idx].points[0].y = -0.1 - human_data[s][m-1][1]
					markerarray_msg.markers[line_idx].points[0].z = human_data[s][m-1][2] - 0.7
					markerarray_msg.markers[line_idx].points[1].x = 0.7 - human_data[s][m][0]
					markerarray_msg.markers[line_idx].points[1].y = -0.1 - human_data[s][m][1]
					markerarray_msg.markers[line_idx].points[1].z = human_data[s][m][2] - 0.7
	
			trajectory_msg.state.joint_state.position = robot_data[s].tolist() + joint_l
		
			for j in range(len(markerarray_msg.markers)):
				markerarray_msg.markers[j].header.stamp = stamp

			trajectory_msg.state.joint_state.header.stamp = stamp

			robot_pub.publish(trajectory_msg)
			human_pub.publish(markerarray_msg)
			rate.sleep()
			if rospy.is_shutdown():
				break
		if rospy.is_shutdown():
			break
