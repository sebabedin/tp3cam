import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
#import sensor_msgs.point_cloud2 as pc2
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class CheCamera(object):

    def __init__(self, node, name='camera'):
        self.info_is_ready = False
        self.node = node
        self.name = name

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create()
    
    def NewCameraInfoMsg(self, camera_info_msg):
        self.info_is_ready = True
        self.d = camera_info_msg.d
        self.k = camera_info_msg.k
        self.r = camera_info_msg.r
        self.p = camera_info_msg.p.reshape((3, 4))
        self.binning_x = camera_info_msg.binning_x
        self.binning_y = camera_info_msg.binning_y
        self.roi = camera_info_msg.roi
        return True

    def NewImageMsg(self, img_msg):
        if (not self.info_is_ready):
            self.node.get_logger().warning(f'{self.name} no ready')
            return False

        self.img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.kp = self.orb.detect(self.img, None)
        self.kp, self.des = self.orb.compute(self.img, self.kp)
        return True
    
    def DrawKeypoints(self):
        return cv2.drawKeypoints(
            self.img,
            self.kp,
            outImage = None,
            color=(255,0,0))

class CheStereoCamera(object):

    CONFIG_FILTER_GOOD_MATCHES = 0.5 # nose, pero venia 0.5 en el ejemplo

    def __init__(self, node):
        self.node = node
        self.left = CheCamera(self.node)
        self.right = CheCamera(self.node)

        self.bridge = CvBridge()
        self.bf = cv2.BFMatcher()

    def Filter_Matches(self):
        self.good_matches_mask = [[0,0] for i in range(len(self.matches))]
        self.good_matches = []
        for i,(m,n) in enumerate(self.matches):
            if m.distance < self.CONFIG_FILTER_GOOD_MATCHES*n.distance:
                self.good_matches_mask[i]=[1,0]
                self.good_matches.append(m)
    
    def TriangulatePoints(self):
        # Get corresponding keypoints in both images
        matched_keypoints_left = [self.left.kp[m.queryIdx] for m in self.good_matches]
        matched_keypoints_right = [self.right.kp[m.trainIdx] for m in self.good_matches]

        # Convert keypoints to numpy arrays
        pts_left = np.float32([kp.pt for kp in matched_keypoints_left])
        pts_right = np.float32([kp.pt for kp in matched_keypoints_right])
        #print(self.left.p)
        self.points_3d = cv2.triangulatePoints(self.left.p, self.right.p, pts_left.T, pts_right.T)
        # Convert to homogeneous coordinates
        self.points_3d /= self.points_3d[3]
        self.points_3d = self.points_3d[:3]
        

    def NewImagesMsg(self, left_img_msg, right_img_msg):
        self.node.get_logger().info('New Images')

        # if (not self.left.info_is_ready) or (not self.right.info_is_ready):
        #     self.node.get_logger().warning('Cameras are not ready')
        #     return False

        if not self.left.NewImageMsg(left_img_msg):
            return False
        
        if not self.right.NewImageMsg(right_img_msg):
            return False

        self.matches = self.bf.knnMatch(
            self.left.des,
            self.right.des,
            k=2)

        self.Filter_Matches()

        self.TriangulatePoints()

        return True

    def DrawMatches(self):
        return cv2.drawMatchesKnn(
            self.left.img, self.left.kp, 
            self.right.img, self.right.kp,
            self.matches,
            outImg=None,
            # matchColor=(0, 255, 0),
            matchesMask=self.good_matches_mask,
            # singlePointColor=(255, 0, 0),
            flags=0)

    def CreatePointCloud(self):

        # Create a PointCloud2 message
        pc_msg = PointCloud2()

        # Set the header information
        pc_msg.header.stamp = self.node.get_clock().now().to_msg()
        pc_msg.header.frame_id = "map"  # Set the appropriate frame_id

        # Define the point cloud fields (x, y, z)
        #fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #        PointField('y', 4, PointField.FLOAT32, 1),
        #        PointField('z', 8, PointField.FLOAT32, 1)]

        # Return the PointCloud2 message
        return pc2.create_cloud_xyz32(pc_msg.header, self.points_3d.transpose())


class TP3Node(Node):
    
    topic_left_camera_info = '/left/camera_info'
    topic_left_image_rect = '/left/image_rect'
    topic_left_kp = '/left/kp'

    topic_right_camera_info = '/right/camera_info'
    topic_right_image_rect = '/right/image_rect'
    topic_right_kp = '/right/kp'

    topic_stereo_matches = '/stereo/matches'
    topic_stereo_pointcloud = '/stereo/pointcloud'

    def __init__(self):
        super().__init__('tp3_node')

        self.get_logger().info('Init node')

        self.bridge = CvBridge()
        self.stereo = CheStereoCamera(self)

        self.sub_left_camera_info = self.create_subscription(CameraInfo, self.topic_left_camera_info, self.on_left_info_cb, 0)
        self.sub_right_camera_info = self.create_subscription(CameraInfo, self.topic_right_camera_info, self.on_right_info_cb, 0)

        sub_left_img_rect = message_filters.Subscriber(self, Image, self.topic_left_image_rect)
        sub_right_img_rect = message_filters.Subscriber(self, Image, self.topic_right_image_rect)
        self.subs_img_rect = [sub_left_img_rect, sub_right_img_rect]
        self.ts_img_rect = message_filters.TimeSynchronizer(self.subs_img_rect, 10)
        self.ts_img_rect.registerCallback(self.on_image_rect_cb)

        self.pub_left_img_kp = self.create_publisher(Image, self.topic_left_kp, 1)
        self.pub_right_img_kp = self.create_publisher(Image, self.topic_right_kp, 1)
        self.pub_stereo_matches = self.create_publisher(Image, self.topic_stereo_matches, 1)
        self.pub_stereo_pointcloud = self.create_publisher(PointCloud2, self.topic_stereo_pointcloud, 1)

        self.get_logger().info('Start node')
    
    def on_left_info_cb(self, msg):
        self.stereo.left.NewCameraInfoMsg(msg)
    
    def on_right_info_cb(self, msg):
        self.stereo.right.NewCameraInfoMsg(msg)

    def on_image_rect_cb(self, left_msg, right_msg):
        if self.stereo.NewImagesMsg(left_msg, right_msg):
            self.PubKeypoints()
            self.PubMatches()
            self.PubPointCloud()

    def GenericPubImg(self, pub, img):
        pub.publish(self.bridge.cv2_to_imgmsg(img))

    def PubKeypoints(self):
        self.GenericPubImg(self.pub_left_img_kp, self.stereo.left.DrawKeypoints())
        self.GenericPubImg(self.pub_right_img_kp, self.stereo.right.DrawKeypoints())
        # self.imgData_left.PubKeypoints()
        # self.imgData_rigth.PubKeypoints()

    def PubMatches(self):
        self.GenericPubImg(self.pub_stereo_matches, self.stereo.DrawMatches())

    def PubPointCloud(self):
        self.pub_stereo_pointcloud.publish(self.stereo.CreatePointCloud())




    # class StereoImgData(object):

    #     class ImgData(object):
    #         LEFT_INDEX = 0
    #         RIGHT_INDEX = 1
            
    #         def __init__(self, cam_index, node, img):
    #             self.node = node
    #             self.cam_index = cam_index
                
    #             self.pub_img_kp = self.node.pub_img_right_kp_ if self.RIGHT_INDEX == cam_index else self.node.pub_img_left_kp_

    #             self.img = img
    #             self.kp = self.node.orb_.detect(self.img, None)
    #             self.kp, self.des = self.node .orb_.compute(self.img, self.kp)
            
    #         def DrawKeypoints(self):
    #             return cv2.drawKeypoints(
    #                 self.img,
    #                 self.kp,
    #                 outImage = None,
    #                 color=(255,0,0))
            
    #         def PubKeypoints(self):
    #             self.node.PubImg(self.pub_img_kp, self.DrawKeypoints())
        
    #     def __init__(self, node, image_left, image_right):
    #         self.node = node
    #         self.imgData_left = self.ImgData(self.ImgData.LEFT_INDEX, node, image_left)
    #         self.imgData_rigth = self.ImgData(self.ImgData.RIGHT_INDEX, node, image_right)

    #         self.matches = self.node.bf_.knnMatch(
    #             self.imgData_left.des,
    #             self.imgData_rigth.des,
    #             k=2)
            
    #         # Need to draw only good matches, so create a mask
    #         self.good_matches = [[0,0] for i in range(len(self.matches))]
            
    #         # ratio test as per Lowe's paper
    #         for i,(m,n) in enumerate(self.matches):
    #             if m.distance < 0.75*n.distance:
    #                 self.good_matches[i]=[1,0]
        
    #     def PubKeypoints(self):
    #         self.imgData_left.PubKeypoints()
    #         self.imgData_rigth.PubKeypoints()
        
    #     def DrawMatches(self):
    #         return cv2.drawMatchesKnn(
    #             self.imgData_left.img, self.imgData_left.kp, 
    #             self.imgData_rigth.img, self.imgData_rigth.kp,
    #             self.matches,
    #             outImg=None,
    #             # matchColor=(0, 255, 0),
    #             matchesMask=self.good_matches,
    #             # singlePointColor=(255, 0, 0),
    #             flags=0)

    #     def PubMatches(self):
    #         self.node.PubImg(self.node.pub_img_matches_, self.DrawMatches())

def main(args=None):
    rclpy.init(args=args)
    node = TP3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()