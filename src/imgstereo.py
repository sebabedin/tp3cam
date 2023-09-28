import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

#    1 from cv_bridge import CvBridge
#    2 bridge = CvBridge()
#    3 cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

class ImgStereoNode(Node):
      
    def __init__(self):
        super().__init__('img_sub')

        self.get_logger().info('Init node')

        # self.left_img = self.create_subscription(Image, 'stereo/left/image_raw', self.on_img_left_cb_, 0)
        # self.right_img = self.create_subscription(Image, 'stereo/right/image_raw', self.on_img_right_cb_, 0)

        self.bridge_ = CvBridge()
        self.fast_ = cv2.FastFeatureDetector()
        # self.orb_ = cv2.ORB()
        self.orb_ = cv2.ORB_create()
        self.bf_ = cv2.BFMatcher()

        self.left_rect = message_filters.Subscriber(self, Image, 'stereo/left/image_raw')
        self.right_rect = message_filters.Subscriber(self, Image, 'stereo/right/image_raw')

        self.pub_img_left_kp_ = self.create_publisher(Image, 'image/left/kp', 1)
        self.pub_img_right_kp_ = self.create_publisher(Image, 'image/right/kp', 1)
        # self.pub_img_kp_l_ = self.create_publisher(Image, 'image/left/kp', 1)
        # self.pub_img_kp_r_ = self.create_publisher(Image, 'image/right/kp', 1)
        self.pub_img_matches_ = self.create_publisher(Image, 'image/matches', 1)

        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)
        ts.registerCallback(self.on_img_cb_)

        self.get_logger().info('Start node')
    
    # def on_img_left_cb_(self, msg):
    #     self.get_logger().info(f'Left: {msg.header.stamp},')

    # def on_img_right_cb_(self, msg):
    #     self.get_logger().info(f'Right: {msg.header.stamp},')

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
            
            # good = []
            # matched_image = cv2.drawMatchesKnn(
            #     img_data_l.img, img_data_l.kp, 
            #     img_data_r.img, img_data_r.kp,
            #     matches, None,
            #     matchColor=(0, 255, 0),
            #     matchesMask=None,
            #     singlePointColor=(255, 0, 0),
            #     flags=0)
        
        def PubKeypoints(self):
            self.imgData_left.PubKeypoints()
            self.imgData_rigth.PubKeypoints()
        
        def DrawMatches(self):
            return cv2.drawMatchesKnn(
                self.imgData_left.img, self.imgData_left.kp, 
                self.imgData_rigth.img, self.imgData_rigth.kp,
                self.matches,
                outImg=None,
                matchColor=(0, 255, 0),
                matchesMask=None,
                singlePointColor=(255, 0, 0),
                flags=0)

        def PubMatches(self):
            self.node.PubImg(self.node.pub_img_matches_, self.DrawMatches())
            # self.left_imgData.PubKeypoints()
            # self.rigth_imgData.PubKeypoints()

    def PubImg(self, pub, img):
        pub.publish(self.bridge_.cv2_to_imgmsg(img))

    def on_img_orb_cb_(self, image_left, image_right):
        stereo_img_data = self.StereoImgData(self, image_left, image_right)
        stereo_img_data.PubKeypoints()
        stereo_img_data.PubMatches()
        # img_data_l = self.ImgData(self.ImgData.LEFT_INDEX, self, image_left)
        # img_data_r = self.ImgData(self.ImgData.RIGHT_INDEX, self, image_right)

        # self.PubImg(self.pub_img_kp_l_, img_data_l.DrawKeypoints())
        # self.PubImg(self.pub_img_kp_r_, img_data_r.DrawKeypoints())

        # # BFMatcher with default parameters
        # bf = cv2.BFMatcher()
        # # finding matches from BFMatcher()
        # matches = self.bf_.knnMatch(img_data_l.des, img_data_r.des, k=2) 
        # # Apply ratio test
        # good = []
        # matched_image = cv2.drawMatchesKnn(
        #     img_data_l.img, img_data_l.kp, 
        #     img_data_r.img, img_data_r.kp,
        #     matches, None,
        #     matchColor=(0, 255, 0),
        #     matchesMask=None,
        #     singlePointColor=(255, 0, 0),
        #     flags=0)
    # # creating a criteria for the good matches
    # # and appending good matchings in good[]
    #     for m, n in matches:
    #         # print("m.distance is <",m.distance,"> 
    #         # 1.001*n.distance is <",0.98*n.distance,">")
    #         if m.distance < 0.98 * n.distance:
    #             good.append([m])
    # # for jupyter notebook use this function
    # # to see output image
    # #   plt.imshow(matched_image)
    
    # # if you are using python then run this-
    #     cv2.imshow("matches", matched_image)
  

        # self.PubImg(self.pub_img_matches_, matched_image)
        # msg = self.bridge_.cv2_to_imgmsg(kp_img)
        # self.pub_img_left_kp_.publish(msg)

        # msg = self.bridge_.cv2_to_imgmsg(image_left)
        # self.get_logger().info('14')
        # self.pub_img_left_kp_.publish(msg)

    def on_img_cb_(self, left_msg, right_msg):
        cv_image_left = self.bridge_.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
        cv_image_right = self.bridge_.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
        self.on_img_orb_cb_(cv_image_left, cv_image_right)

#         fast = cv2.FastFeatureDetector()
#         # find and draw the keypoints
#         kp = fast.detect(img,None)
#         img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

#         self.get_logger().info(f'Sync, left: {left_msg.header.stamp}, right: {right_msg.header.stamp},')

def main(args=None):
    rclpy.init(args=args)
    node = ImgStereoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()