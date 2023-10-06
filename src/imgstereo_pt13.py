import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
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
        self.k = camera_info_msg.k.reshape((3, 3))
        self.r = camera_info_msg.r
        self.p = camera_info_msg.p.reshape((3, 4))
        self.binning_x = camera_info_msg.binning_x
        self.binning_y = camera_info_msg.binning_y
        self.roi = camera_info_msg.roi
        # self.matrix = self.p[:, :3]
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
    
    def FilterKeyPoints(self, idxs):
        self.points = np.float32([self.kp[idx].pt for idx in idxs])

class CheStereoCamera(object):

    CONFIG_FILTER_GOOD_MATCHES = 0.5 # nose, pero venia 0.5 en el ejemplo
    MIN_MATCH_COUNT = 10

    def __init__(self, node):
        self.first_sterep_image = True
        self.node = node
        self.left = CheCamera(self.node)
        self.right = CheCamera(self.node)

        self.bridge = CvBridge()
        self.bf = cv2.BFMatcher()

    def NewImagesMsg(self, left_img_msg, right_img_msg):
        if not self.left.NewImageMsg(left_img_msg):
            return False
        
        if not self.right.NewImageMsg(right_img_msg):
            return False

        if self.first_sterep_image:
            # self.first_sterep_image = False
            self.node.get_logger().info(f'Camera left P: {self.left.p}')
            self.node.get_logger().info(f'Camera right P: {self.right.p}')
            
            Tx = self.right.p[0, 3]
            fx = self.right.p[0, 0]
            self.B = Tx / -fx
            # self.left.pTx = -fx' * B
            self.node.get_logger().info(f'Stereo B: {self.B}')

        self.FindMatches()
        self.Triangulate()

        if not self.FindHomography():
            self.node.get_logger().warn('Homography Fault')
            return False
        
        self.DisparityMap()
        self.Reproject()
        self.FindPoseBetweenCameras()
        
        if self.first_sterep_image:
            self.node.get_logger().info(f'Stereo Camera Right Pose: {self.camera_pose}')

        self.first_sterep_image = False
        return True

    def FindMatches(self):
        self.matches = self.bf.knnMatch(
            self.left.des,
            self.right.des,
            k=2)

        self.good_matches_mask = [[0,0] for i in range(len(self.matches))]
        self.good_matches = []
        
        for i,(m,n) in enumerate(self.matches):
            if m.distance < self.CONFIG_FILTER_GOOD_MATCHES*n.distance:
                self.good_matches_mask[i]=[1,0]
                self.good_matches.append(m)
        
        self.left.FilterKeyPoints([m.queryIdx for m in self.good_matches])
        self.right.FilterKeyPoints([m.trainIdx for m in self.good_matches])

    def DrawMatches(self):
        return cv2.drawMatchesKnn(
            self.left.img, self.left.kp, 
            self.right.img, self.right.kp,
            self.matches,
            outImg=None,
            matchesMask=self.good_matches_mask,
            flags=0)

    def Triangulate(self):
        self.points_3d = cv2.triangulatePoints(self.left.p, self.right.p, self.left.points.T, self.right.points.T)
        self.points_3d /= self.points_3d[3]
        self.points_3d = self.points_3d[:3]

    def FindHomography(self):
        if len(self.good_matches) > self.MIN_MATCH_COUNT:

            self.homography, mask = cv2.findHomography(self.left.points, self.right.points, cv2.RANSAC,5.0)
            
            self.ransac_mask = mask.ravel().tolist()
            return True
        else:
            self.node.get_logger().warn( "Not enough matches are found - {}/{}".format(len(self.good_matches), self.MIN_MATCH_COUNT) )    
            return False
    
    def DrawHomography(self):
        dst = cv2.perspectiveTransform(self.left.points.reshape(-1,1,2),self.homography)

        points_1 = self.left.points.reshape(-1,2)
        points_2 = dst.reshape(-1,2)
        points_1 = points_1[self.ransac_mask]
        points_2 = points_2[self.ransac_mask]
        
        for pt1, pt2 in zip(points_1, points_2):
            cv2.circle(self.left.img, pt1.astype(int), 5, (0, 0, 255), -1)  
            cv2.circle(self.right.img, pt2.astype(int), 5, (0, 0, 255), -1) 

        h,w = self.left.img.shape
        result_image = np.zeros((h, w*2), dtype=np.uint8)
        # Concatenate the two images side by side
        result_image[:, :w] = self.left.img
        result_image[:, w:] = self.right.img
        # Draw lines connecting corresponding points
        for pt1, pt2 in zip(points_1, points_2):
            pt1 = (int(pt1[0]), int(pt1[1]))  
            pt2 = (int(pt2[0]) + w, int(pt2[1]))  
            cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)
        
        return result_image

    def DisparityMap(self):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,          # Minimum disparity value (usually 0)
            numDisparities=64,       # Number of disparity levels
            blockSize=3,             # Block size for matching
            P1=8 * 1 * 3**2,         # Penalty 1 parameter
            P2=32 * 1 * 3**2,        # Penalty 2 parameter
            disp12MaxDiff=1,         # Maximum allowed difference in the left-right disparity
            preFilterCap=63,         # Pre-filter cap
            uniquenessRatio=10,      # Uniqueness ratio
            speckleWindowSize=100,   # Speckle window size
            speckleRange=1,         # Speckle range
            #mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.disparity = stereo.compute(self.left.img, self.right.img)
        self.disparity = cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return True

    def Reproject(self):
        # Confección de Q:
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
        cx1 = self.left.p[0,2]
        cx2 = self.right.p[0,2]
        cy = self.left.p[1,2]
        f = self.right.p[0,0]
        Tx = self.right.p[0,3] / f
        Q = np.float32([[1, 0, 0, -cx1],
                        [0, 1, 0, -cy], 
                        [0, 0, 0, f],  
                        [0, 0, -1/Tx, (cx1-cx2)/Tx]])

        self.reproject_points = cv2.reprojectImageTo3D(self.disparity, Q)

    def FindPoseBetweenCameras(self):
        self.E, mask = cv2.findEssentialMat(self.left.points, self.right.points, self.left.k)
        points, self.R_est, self.t_est, mask = cv2.recoverPose(self.E, self.left.points, self.right.points)
        self.camera_pose = np.concatenate((self.R_est, self.t_est*self.B), axis=1)

class TP3Node(Node):
    
    topic_left_camera_info = '/left/camera_info'
    topic_left_image_rect = '/left/image_rect'
    topic_left_kp = '/left/kp'

    topic_right_camera_info = '/right/camera_info'
    topic_right_image_rect = '/right/image_rect'
    topic_right_kp = '/right/kp'

    topic_stereo_matches = '/stereo/matches'
    topic_stereo_pointcloud = '/stereo/pointcloud'
    topic_stereo_homography = '/stereo/homography'
    topic_stereo_disparity = '/stereo/disparity'
    topic_stereo_reproject = '/stereo/reproject'

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
        self.pub_stereo_homography = self.create_publisher(Image, self.topic_stereo_homography, 1)
        self.pub_stereo_disparity = self.create_publisher(Image, self.topic_stereo_disparity, 1)
        self.pub_stereo_reproject = self.create_publisher(PointCloud2, self.topic_stereo_reproject, 1)

        self.get_logger().info('Start node')
    
    def on_left_info_cb(self, msg):
        self.stereo.left.NewCameraInfoMsg(msg)
    
    def on_right_info_cb(self, msg):
        self.stereo.right.NewCameraInfoMsg(msg)

    def on_image_rect_cb(self, left_msg, right_msg):
        if self.stereo.NewImagesMsg(left_msg, right_msg):
            self.GenericPubImg(self.pub_left_img_kp, self.stereo.left.DrawKeypoints())
            self.GenericPubImg(self.pub_right_img_kp, self.stereo.right.DrawKeypoints())
            self.GenericPubImg(self.pub_stereo_matches, self.stereo.DrawMatches())
            self.GenericPubPointCloud(self.pub_stereo_reproject, self.stereo.points_3d.transpose())
            self.GenericPubImg(self.pub_stereo_homography, self.stereo.DrawHomography())
            self.GenericPubImg(self.pub_stereo_disparity, self.stereo.disparity)
            self.GenericPubPointCloud(self.pub_stereo_reproject, self.stereo.reproject_points)

    def GenericPubImg(self, pub, img):
        pub.publish(self.bridge.cv2_to_imgmsg(img))
    
    def GenericPubPointCloud(self, pub, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map" 
        pub.publish(point_cloud2.create_cloud_xyz32(msg.header, points))

def main(args=None):
    rclpy.init(args=args)
    node = TP3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()