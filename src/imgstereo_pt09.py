import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class ImgStereoNode(Node):
      
    def __init__(self):
        super().__init__('img_sub')

        self.get_logger().info('Init node')

        self.bridge_ = CvBridge()
        self.fast_ = cv2.FastFeatureDetector()
        self.orb_ = cv2.ORB_create()
        self.bf_ = cv2.BFMatcher()

        self.left_rect = message_filters.Subscriber(self, Image, '/left/image_rect')
        self.right_rect = message_filters.Subscriber(self, Image, '/right/image_rect')

        self.pub_img_left_kp_ = self.create_publisher(Image, 'image/left/kp', 1)
        self.pub_img_right_kp_ = self.create_publisher(Image, 'image/right/kp', 1)
        self.pub_img_matches_ = self.create_publisher(Image, 'image/matches', 1)

        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)
        ts.registerCallback(self.on_img_cb_)

        self.get_logger().info('Start node')
    
    class StereoImgData(object):

        class ImgData(object):
            LEFT_INDEX = 0
            RIGHT_INDEX = 1
            
            def __init__(self, cam_index, node, img):
                self.node = node
                self.cam_index = cam_index
                
                self.pub_img_kp = self.node.pub_img_right_kp_ if self.RIGHT_INDEX == cam_index else self.node.pub_img_left_kp_

                self.img = img
                self.kp = self.node.orb_.detect(self.img, None)
                self.kp, self.des = self.node .orb_.compute(self.img, self.kp)
            
            def DrawKeypoints(self):
                return cv2.drawKeypoints(
                    self.img,
                    self.kp,
                    outImage = None,
                    color=(255,0,0))
            
            def PubKeypoints(self):
                self.node.PubImg(self.pub_img_kp, self.DrawKeypoints())
        
        def __init__(self, node, image_left, image_right):
            self.node = node
            self.imgData_left = self.ImgData(self.ImgData.LEFT_INDEX, node, image_left)
            self.imgData_rigth = self.ImgData(self.ImgData.RIGHT_INDEX, node, image_right)

            self.matches = self.node.bf_.knnMatch(
                self.imgData_left.des,
                self.imgData_rigth.des,
                k=2)
            
            # Need to draw only good matches, so create a mask
            self.good_matches = [[0,0] for i in range(len(self.matches))]
            
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(self.matches):
                if m.distance < 0.75*n.distance:
                    self.good_matches[i]=[1,0]
        
        def PubKeypoints(self):
            self.imgData_left.PubKeypoints()
            self.imgData_rigth.PubKeypoints()
        
        def DrawMatches(self):
            return cv2.drawMatchesKnn(
                self.imgData_left.img, self.imgData_left.kp, 
                self.imgData_rigth.img, self.imgData_rigth.kp,
                self.matches,
                outImg=None,
                # matchColor=(0, 255, 0),
                matchesMask=self.good_matches,
                # singlePointColor=(255, 0, 0),
                flags=0)

        def PubMatches(self):
            self.node.PubImg(self.node.pub_img_matches_, self.DrawMatches())

    def PubImg(self, pub, img):
        pub.publish(self.bridge_.cv2_to_imgmsg(img))

    def on_img_orb_cb_(self, image_left, image_right):
        stereo_img_data = self.StereoImgData(self, image_left, image_right)
        stereo_img_data.PubKeypoints()
        stereo_img_data.PubMatches()

        # X = cv2.triangulatePoints(P1[:3], P2[:3], x1, x2)

    def on_img_cb_(self, left_msg, right_msg):
        cv_image_left = self.bridge_.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
        cv_image_right = self.bridge_.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
        self.on_img_orb_cb_(cv_image_left, cv_image_right)

def main(args=None):
    rclpy.init(args=args)
    node = ImgStereoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()