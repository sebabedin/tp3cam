import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.02 --ros-args -r image:=/camera/left/image_raw -p camera:=/camera_left
# ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.02 --ros-args -r image:=/camera/right/image_raw -p camera:=/camera_right

CALIBRATION_PATH = '/home/seb/ros2ws/rbtp3_ws/src/tp3cam/resource/calibrationdata'

class ImgMinimalPubNode(Node):
      
      def __init__(self):
         super().__init__('img_pub')
         
         self.pub_img_left_ = self.create_publisher(Image, 'camera/left/image_raw', 1)
         self.pub_img_right_ = self.create_publisher(Image, 'camera/right/image_raw', 1)

         timer_period = 0.5
         self.timer_ = self.create_timer(timer_period, self.timer_callback_)
         
         self.index_ = 0
         
         self.bridge_ = CvBridge()

      def timer_callback_(self):
         file_name = 'left-%04d.png' % (self.index_,)
         img_file_path = os.path.join(CALIBRATION_PATH, file_name)
         self.cv_image_ = cv2.imread(img_file_path)
         self.pub_img_left_.publish(self.bridge_.cv2_to_imgmsg(np.array(self.cv_image_), "bgr8"))

         file_name = 'right-%04d.png' % (self.index_,)
         img_file_path = os.path.join(CALIBRATION_PATH, file_name)
         self.cv_image_ = cv2.imread(img_file_path)
         self.pub_img_right_.publish(self.bridge_.cv2_to_imgmsg(np.array(self.cv_image_), "bgr8"))

         self.index_ += 1
         if 114 < self.index_:
             self.index_ = 0
         self.get_logger().info('Publishing an image')

def main(args=None):
    rclpy.init(args=args)
    node = ImgMinimalPubNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()