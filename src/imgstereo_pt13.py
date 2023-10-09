import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
#import sensor_msgs.point_cloud2 as pc2
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import transforms3d.quaternions as quaternions
import tf2_ros as tf

class Frame(object):
    def __init__(self, img, kp, des):
        self.img = img
        self.kp = kp
        self.des = des

class CheCamera(object):

    def __init__(self, node, name='camera'):
        self.info_is_ready = False
        self.node = node
        self.name = name

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create()
        self.prev_frame = None
        self.first_frame = True
    
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

        if self.first_frame:
            self.first_frame = False
        else:
            self.prev_frame = self.frame

        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)
        self.frame = Frame(img, kp, des)
        return True
    
    def DrawKeypoints(self):
        return cv2.drawKeypoints(
            self.frame.img,
            self.frame.kp,
            outImage = None,
            color=(255,0,0))

class CheStereoCamera(object):

    CONFIG_FILTER_GOOD_MATCHES = 0.5 # nose, pero venia 0.5 en el ejemplo
    MIN_MATCH_COUNT = 10

    def __init__(self, node):
        self.node = node
        self.left = CheCamera(self.node)
        self.right = CheCamera(self.node)

        self.bridge = CvBridge()
        self.bf = cv2.BFMatcher()
        self.R_odom = np.eye(3)
        self.t_odom = np.zeros(3)

    def Filter_Matches(self, matches):
        good_matches_mask = [[0,0] for i in range(len(matches))]
        good_matches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < self.CONFIG_FILTER_GOOD_MATCHES*n.distance:
                good_matches_mask[i]=[1,0]
                good_matches.append(m)
        
        return good_matches, good_matches_mask
    
    def TriangulatePoints(self):
        # Get corresponding keypoints in both images and convert keypoints to numpy arrays
        self.pts_left = np.float32([self.left.frame.kp[m.queryIdx].pt for m in self.good_matches])
        self.pts_right = np.float32([self.right.frame.kp[m.trainIdx].pt for m in self.good_matches])

        self.points_3d = cv2.triangulatePoints(self.left.p, self.right.p, self.pts_left.T, self.pts_right.T)
        # Convert to homogeneous coordinates
        self.points_3d /= self.points_3d[3]
        self.points_3d = self.points_3d[:3]
        
    def FindHomography(self):
        if len(self.good_matches) > self.MIN_MATCH_COUNT:
            self.homography, mask = cv2.findHomography(self.pts_left, self.pts_right, cv2.RANSAC,5.0)
            self.ransac_mask = mask.ravel()
            return True
        else:
            self.node.get_logger().warn( "Not enough matches are found - {}/{}".format(len(self.good_matches), self.MIN_MATCH_COUNT) )    
            return False
    
    def DrawHomography(self):
        dst = cv2.perspectiveTransform(self.pts_left.reshape(-1,1,2),self.homography)

        points_1 = self.pts_left.reshape(-1,2)
        points_2 = dst.reshape(-1,2)
        points_1 = points_1[self.ransac_mask == 1]
        points_2 = points_2[self.ransac_mask == 1]
        for pt1, pt2 in zip(points_1, points_2):
            cv2.circle(self.left.frame.img, pt1.astype(int), 5, (0, 0, 255), -1)  
            cv2.circle(self.right.frame.img, pt2.astype(int), 5, (0, 0, 255), -1) 

        h,w = self.left.frame.img.shape
        result_image = np.zeros((h, w*2), dtype=np.uint8)
        # Concatenate the two images side by side
        result_image[:, :w] = self.left.frame.img
        result_image[:, w:] = self.right.frame.img
        # Draw lines connecting corresponding points
        for pt1, pt2 in zip(points_1, points_2):
            pt1 = (int(pt1[0]), int(pt1[1]))  
            pt2 = (int(pt2[0]) + w, int(pt2[1]))  
            cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)
        
        return result_image

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
            self.left.frame.des,
            self.right.frame.des,
            k=2)

        self.good_matches, self.good_matches_mask = self.Filter_Matches(self.matches)

        self.TriangulatePoints()
 
        if not self.FindHomography():
            return False

        self.ComputeDisparity()

        self.Reconstruction3D()

        #R, t = self.MonocularPoseEstimation(self.pts_left, self.pts_right, self.left.k.reshape(3,3), self.right.k.reshape(3,3), normalize=False)
        #f = self.right.p[0,0]
        #Tx = self.right.p[0,3] / f
        #print(R)
        #print(Tx*t)
        if not self.MonocularVisualOdometry():
            return False
        
        return True

    def DrawMatches(self):
        return cv2.drawMatchesKnn(
            self.left.frame.img, self.left.frame.kp, 
            self.right.frame.img, self.right.frame.kp,
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

        # Return the PointCloud2 message
        return pc2.create_cloud_xyz32(pc_msg.header, self.points_3d.transpose())

    def ComputeDisparity(self):
        # Create an SGBM StereoMatcher object
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
        self.disparity_map = stereo.compute(self.left.frame.img, self.right.frame.img)
        self.disparity_map = cv2.normalize(self.disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def Reconstruction3D(self):
        cx1 = self.left.p[0,2]
        cx2 = self.right.p[0,2]
        cy = self.left.p[1,2]
        f = self.right.p[0,0]
        Tx = self.right.p[0,3] / f
        Q = np.float32([[1, 0, 0, -cx1],
                        [0, 1, 0, -cy], 
                        [0, 0, 0, f],  
                        [0, 0, -1/Tx, (cx1-cx2)/Tx]])
        self.reconstructed_points = cv2.reprojectImageTo3D(self.disparity_map, Q)
    
    def DrawReconstruction(self):
        # Create a PointCloud2 message
        pc_msg = PointCloud2()

        # Set the header information
        pc_msg.header.stamp = self.node.get_clock().now().to_msg()
        pc_msg.header.frame_id = "map"  # Set the appropriate frame_id

        # Return the PointCloud2 message
        return pc2.create_cloud_xyz32(pc_msg.header, self.reconstructed_points)
    
    def MonocularPoseEstimation(self, pts_1, pts_2, K_1, K_2=None, normalize=False):
        if normalize:
            # Normalize for Esential Matrix calculation
            pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_1, axis=1), cameraMatrix=K_1, distCoeffs=None)
            pts_r_norm = cv2.undistortPoints(np.expand_dims(pts_2, axis=1), cameraMatrix=K_2, distCoeffs=None)
            essential_matrix, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            points, R, t, mask = cv2.recoverPose(essential_matrix, pts_l_norm, pts_r_norm)
            return R,t.reshape(3)
        else:
            essential_matrix, mask = cv2.findEssentialMat(pts_1, pts_2, K_1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            retval, R, t, mask = cv2.recoverPose(essential_matrix, pts_1, pts_2, mask=mask)
            return R,t.reshape(3)

    def MonocularVisualOdometry(self):
        if self.left.prev_frame is not None:
            odom_matches = self.bf.knnMatch(
                self.left.prev_frame.des,
                self.left.frame.des,
                k=2)
            

            pts_prev_left = np.float32([self.left.prev_frame.kp[m.queryIdx].pt for m,_ in odom_matches])#good_odom_matches])
            pts_curr_left = np.float32([self.left.frame.kp[m.trainIdx].pt for m,_ in odom_matches])#good_odom_matches])
            R_odom, t_odom = self.MonocularPoseEstimation(pts_prev_left, pts_curr_left, self.left.k.reshape(3,3))
            R_odom = R_odom.transpose()
            t_odom = np.dot(-R_odom.transpose(), t_odom)
            t_odom *= 0.2 # scale?
            self.t_odom = np.dot(self.R_odom, t_odom) + self.t_odom
            self.R_odom = np.dot(self.R_odom, R_odom)
            return True

        return False
    
    def GetPoseMsg(self):
        q_odom = quaternions.mat2quat(self.R_odom)
        pose_msg = Pose()        
        pose_msg.position.x = self.t_odom[0]
        pose_msg.position.y = self.t_odom[1]
        pose_msg.position.z = self.t_odom[2]
        pose_msg.orientation.w = q_odom[0]
        pose_msg.orientation.x = q_odom[1]
        pose_msg.orientation.y = q_odom[2]
        pose_msg.orientation.z = q_odom[3]
        return pose_msg
    
    def GetOdometryMsg(self):
        odometry_msg = Odometry()
        odometry_msg.header.stamp = self.node.get_clock().now().to_msg()
        odometry_msg.header.frame_id = 'odom'
        pose = self.GetPoseMsg()
        odometry_msg.pose.pose = pose
        return odometry_msg



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
    topic_disparity_map = '/stereo/disparity_map'
    
    topic_reconstruction = '/stereo/reconstruction'

    topic_pose = '/pose'
    
    topic_odom = '/odom'

    topic_path = '/path'

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
        self.pub_disparity_map = self.create_publisher(Image, self.topic_disparity_map, 1)
        self.pub_reconstruction = self.create_publisher(PointCloud2, self.topic_reconstruction, 1)
        #self.pub_pose = self.create_publisher(PoseStamped, self.topic_pose, 1)
        self.pub_odometry = self.create_publisher(Odometry, self.topic_odom, 1)
        self.pub_path = self.create_publisher(Path, self.topic_path, 1)
        self.path = Path()
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
            self.PubStereoHomography()
            self.PubDisparityMap()
            self.PubReconstruction()
            self.PubEstimatedPose()

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

    def PubStereoHomography(self):
        self.GenericPubImg(self.pub_stereo_homography, self.stereo.DrawHomography())

    def PubDisparityMap(self):
        self.GenericPubImg(self.pub_disparity_map, self.stereo.disparity_map)

    def PubReconstruction(self):
        self.pub_reconstruction.publish(self.stereo.DrawReconstruction())

    def PubEstimatedPose(self):
        odometry_msg = self.stereo.GetOdometryMsg()        
        self.pub_odometry.publish(odometry_msg)
        pose_msg = PoseStamped()
        pose_msg.header = odometry_msg.header
        pose_msg.pose = odometry_msg.pose.pose
        self.path.poses.append(pose_msg)
        self.pub_path.publish(self.path)

def main(args=None):
    rclpy.init(args=args)
    node = TP3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()