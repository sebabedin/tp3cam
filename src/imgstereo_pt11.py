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
    MIN_MATCH_COUNT = 10

    def __init__(self, node):
        self.first_sterep_image = True
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

    def Triangulate(self):
        # Get corresponding keypoints in both images and convert keypoints to numpy arrays
        self.pts_left = np.float32([self.left.kp[m.queryIdx].pt for m in self.good_matches])
        self.pts_right = np.float32([self.right.kp[m.trainIdx].pt for m in self.good_matches])

        self.points_3d = cv2.triangulatePoints(self.left.p, self.right.p, self.pts_left.T, self.pts_right.T)
        # Convert to homogeneous coordinates
        self.points_3d /= self.points_3d[3]
        self.points_3d = self.points_3d[:3]


        # matches = self.good_matches

        # kp1_idx = np.array([match.queryIdx for match in matches])
        # kp2_idx = np.array([match.trainIdx for match in matches])

        # kp1 = np.array([self.left.kp[idx].pt for idx in kp1_idx])
        # kp2 = np.array([self.right.kp[idx].pt for idx in kp2_idx])

        # kp1_3D = np.ones((3, kp1.shape[0]))
        # kp2_3D = np.ones((3, kp2.shape[0]))

        # kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
        # kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()

        # T_1w = self.left.p
        # T_2w = self.right.p

        # X = cv2.triangulatePoints(self.left.p, self.right.p, kp1.T, kp2.T)
        # X /= X[3]
        # X1 = X[:3]
        # self.points = X1.T

    def FindHomography(self):
        if len(self.good_matches) > self.MIN_MATCH_COUNT:

            self.homography, mask = cv2.findHomography(self.pts_left, self.pts_right, cv2.RANSAC,5.0)
            # print(self.homography)

            # left = np.array([list(x.pt) for x in self.left.kp])
            # right = np.array([list(x.pt) for x in self.right.kp])
            # self.homography, mask = cv2.findHomography(left, right, cv2.RANSAC,5.0)
            # print(self.homography)
            
            self.ransac_mask = mask.ravel().tolist()
            return True
        else:
            self.node.get_logger().warn( "Not enough matches are found - {}/{}".format(len(self.good_matches), self.MIN_MATCH_COUNT) )    
            return False
    
    def DrawHomography(self):
        dst = cv2.perspectiveTransform(self.pts_left.reshape(-1,1,2),self.homography)

        points_1 = self.pts_left.reshape(-1,2)
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

    def NewImagesMsg(self, left_img_msg, right_img_msg):
        if not self.left.NewImageMsg(left_img_msg):
            return False
        
        if not self.right.NewImageMsg(right_img_msg):
            return False


        self.matches = self.bf.knnMatch(
            self.left.des,
            self.right.des,
            k=2)

        self.Filter_Matches()
        self.Triangulate()
        self.DisparityMap()
        
        if not self.FindHomography():
            self.node.get_logger().warn('Homography Fault')
            return False

        
        if self.first_sterep_image:
            self.first_sterep_image = False
            self.node.get_logger().info(f'camera info self.left.p {self.left.p}')
            self.node.get_logger().info(f'camera info self.right.p {self.right.p}')
        
        return True

    def DisparityMap(self):
        # left_img8 = (self.left.img/256).astype('uint8')
        # right_img8 = (self.right.img/256).astype('uint8')
        # left_img8 = cv.cvtColor(self.left.img, code)
        # right_img8 = cv.cvtColor(self.left.img, code)

        # img = self.left.img
        # img *= 1./255;
        # left_img8 = cv2.cvtColor(self.left.img, cv2.CV_8U)
        # right_img8 = cv2.cvtColor(self.right.img, cv2.CV_8U)

        left_img8 = self.left.img
        right_img8 = self.right.img

        # stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)

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

        self.disparity = stereo.compute(left_img8, right_img8)
# /        self.disparity = cv::normalize(self.disparity, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        self.disparity = cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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

        # Return the PointCloud2 message
        return point_cloud2.create_cloud_xyz32(pc_msg.header, self.points_3d.transpose())

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
            self.PubDispatiry()

    def GenericPubImg(self, pub, img):
        pub.publish(self.bridge.cv2_to_imgmsg(img))

    def PubKeypoints(self):
        self.GenericPubImg(self.pub_left_img_kp, self.stereo.left.DrawKeypoints())
        self.GenericPubImg(self.pub_right_img_kp, self.stereo.right.DrawKeypoints())

    def PubMatches(self):
        self.GenericPubImg(self.pub_stereo_matches, self.stereo.DrawMatches())
    
    def PubPointCloud(self):
        self.pub_stereo_pointcloud.publish(self.stereo.CreatePointCloud())
    
    def PubDispatiry(self):
        self.GenericPubImg(self.pub_stereo_disparity, self.stereo.disparity)

    # def PubPointCloud(self):
    #     header = Header()
    #     header.frame_id = 'map'
    #     dtype = PointField.FLOAT32
    #     point_step = 16
    #     fields = [
    #         PointField(name='x', offset=0, datatype=dtype, count=1),
    #         PointField(name='y', offset=4, datatype=dtype, count=1),
    #         PointField(name='z', offset=8, datatype=dtype, count=1),
    #         ]
    #     pc2_msg = point_cloud2.create_cloud(header, fields, self.stereo.points_3d)

    #     self.pub_stereo_pointcloud.publish(pc2_msg)
    
    def PubStereoHomography(self):
        self.GenericPubImg(self.pub_stereo_homography, self.stereo.DrawHomography())

    def PubDisparity(self):
        self.GenericPubImg(self.pub_stereo_disparity, self.stereo.disparity)

def main(args=None):
    rclpy.init(args=args)
    node = TP3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()