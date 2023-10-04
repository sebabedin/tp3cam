import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class ImgFilterSubNode(Node):
      
    def __init__(self):
        super().__init__('img_sub')

        self.get_logger().info('Init node')

        self.left_img = self.create_subscription(Image, 'stereo/left/image_raw', self.on_img_left_cb_, 0)
        self.right_img = self.create_subscription(Image, 'stereo/right/image_raw', self.on_img_right_cb_, 0)

        self.left_rect = message_filters.Subscriber(self, Image, 'stereo/left/image_raw')
        self.right_rect = message_filters.Subscriber(self, Image, 'stereo/right/image_raw')

        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)
        ts.registerCallback(self.on_img_cb_)

        self.get_logger().info('Start node')
    
    def on_img_left_cb_(self, msg):
        self.get_logger().info(f'Left: {msg.header.stamp},')

    def on_img_right_cb_(self, msg):
        self.get_logger().info(f'Right: {msg.header.stamp},')

    def on_img_cb_(self, left_msg, right_msg):
        self.get_logger().info(f'Sync, left: {left_msg.header.stamp}, right: {right_msg.header.stamp},')

def main(args=None):
    rclpy.init(args=args)
    node = ImgFilterSubNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()