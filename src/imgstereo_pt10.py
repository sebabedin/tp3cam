import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class CheCamera(object):

    def __init__(self, node, topic_camera_info='', name=''):
        self.info_is_ready = False
        self.node = node

        self.bridge_ = CvBridge()
        self.orb_ = cv2.ORB_create()

        if '' != topic_camera_info:
            self.sub_info = self.create_subscription(Image, topic_camera_info, self.on_info_cb_, 0)
    
    def on_info_cb_(self, msg):
        if not self.info_is_ready:
            self.NewCameraInfoMsg(msg)

    def NewCameraInfoMsg(self, camera_info_msg):
        self.info_is_ready = False
        self.d = camera_info_msg.d
        self.k = camera_info_msg.k
        self.r = camera_info_msg.r
        self.p = camera_info_msg.p.reshape((3, 4))
        self.binning_x = camera_info_msg.binning_x
        self.binning_y = camera_info_msg.binning_y
        self.roi = camera_info_msg.roi

    def NewImageMsg(self, img_msg):
        self.img = self.bridge_.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.kp = self.orb_.detect(self.img, None)
        self.kp, self.des = self.orb_.compute(self.img, self.kp)

class CheStereoCamera(object):

    TOPIC_CAMERA_INFO = [
        '/left/camera_info',
        '/right/camera_info',
        ]
    
    TOPIC_CAMERA_IMAGE = [
        '/left/image_rect',
        '/right/image_rect',
        ]

    def __init__(self, node):
        self.node = node
        self.cameras = [
            CheCamera(self.node, self.TOPIC_CAMERA_INFO[0]),
            CheCamera(self.node, self.TOPIC_CAMERA_INFO[1]),
            ]
        self.bf_ = cv2.BFMatcher()

        sub_left_img = message_filters.Subscriber(self, Image, self.TOPIC_CAMERA_IMAGE[0])
        sub_right_img = message_filters.Subscriber(self, Image, self.TOPIC_CAMERA_IMAGE[1])
        self.subts_img = message_filters.TimeSynchronizer([sub_left_img, sub_right_img], 10)
        self.subts_img.registerCallback(self.on_image_cb_)

    def NewImagesMsg(self, left_img_msg, right_img_msg):
        self.left_camera.NewImageMsg(left_img_msg)
        self.right_camera.NewImageMsg(right_img_msg)

        self.matches = self.bf_.knnMatch(
            self.left_camera.des,
            self.right_camera.des,
            k=2)
    
    def on_image_cb_(self, left_msg, right_msg):
        if self.STATE_WAITING_IMAGE_RECT == self.state:
            self.state = self.STATE_DO_SOMETHING

            self.stereo_camera = CheStereoCamera(self.left_camera, self.right_camera)
            self.stereo_camera.NewImagesMsg(left_msg, right_msg)

            self.DoSomething()

class TP3Node(Node):
      
    def __init__(self):
        super().__init__('tp3_node')

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

        sub_left_camera_info = message_filters.Subscriber(self, CameraInfo, '/left/camera_info')
        sub_right_camera_info = message_filters.Subscriber(self, CameraInfo, '/right/camera_info')
        self.subs_camera_info = [sub_left_camera_info, sub_right_camera_info]
        self.ts_camera_info = message_filters.TimeSynchronizer(self.subs_camera_info, 10)
        self.ts_camera_info.registerCallback(self.on_camera_info_cb_)

        sub_left_img_rect = message_filters.Subscriber(self, Image, '/left/image_rect')
        sub_right_img_rect = message_filters.Subscriber(self, Image, '/right/image_rect')
        self.subs_img_rect = [sub_left_img_rect, sub_right_img_rect]
        self.ts_img_rect = message_filters.TimeSynchronizer(self.subs_img_rect, 10)
        self.ts_img_rect.registerCallback(self.on_image_rect_cb_)

        self.pub_pcl_ = self.create_publisher(PointCloud2, '/left/pointcloud', 1)


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