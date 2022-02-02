#!/usr/bin/env python3
import rospy
import sys

sys.path.append(r'/home/pi/rospicar_ws/src/picar/picar-x/lib')
from picarx import Picarx
import time
from std_msgs.msg import Int32


def forward(speed):
    px.forward(speed)


def car_angle(msg):
    px.set_dir_servo_angle(msg.data)
    rospy.loginfo(msg.data)


def car_speed(msg):
    px.forward(msg.data)
    rospy.loginfo(msg.data)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/angle", Int32, car_angle)
    rospy.Subscriber("/speed", Int32, car_speed)
    rospy.spin()


if __name__ == '__main__':
    try:
        px = Picarx()
        px.set_camera_servo2_angle(-20)
        listener()
    finally:
        px.set_camera_servo2_angle(0)
        px.stop()
