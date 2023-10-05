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
        # matches = self.matches[self.good_matches]
        # matches = self.matches
        # matches = np.array([x[0] for x in self.matches])
        # filter = np.array([x[0] for x in self.good_matches])
        # nmax = 3 if 3 < len(self.good_matches) else len(self.good_matches)
        # if nmax < len(self.good_matches)
        matches = self.good_matches

        kp1_idx = np.array([match.queryIdx for match in matches])
        kp2_idx = np.array([match.trainIdx for match in matches])

        kp1 = np.array([self.left.kp[idx].pt for idx in kp1_idx])
        kp2 = np.array([self.right.kp[idx].pt for idx in kp2_idx])

        kp1_3D = np.ones((3, kp1.shape[0]))
        kp2_3D = np.ones((3, kp2.shape[0]))

        kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
        kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()

        T_1w = self.left.p
        T_2w = self.right.p

        # X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
        X = cv2.triangulatePoints(self.left.p, self.right.p, kp1.T, kp2.T)
        X /= X[3]
        # X1 = T_1w[:3] @ X
        X1 = X[:3]
        self.points = X1.T
        # self.points = X[:1].T
        # X2 = T_2w[:3] @ X

        # # msg = array_to_pointcloud2(X1)

        # self.header = Header()
        # self.header.frame_id = 'map'


        # dtype = PointField.FLOAT32
        # point_step = 16
        # self.fields = [
        #     PointField(name='x', offset=0, datatype=dtype, count=1),
        #     PointField(name='y', offset=4, datatype=dtype, count=1),
        #     PointField(name='z', offset=8, datatype=dtype, count=1),
        #     ]
        # pc2_msg = point_cloud2.create_cloud(self.header, self.fields, points)

        # # pc2_msg = point_cloud2.create_cloud(self.header, self.fields, points)
        # self.pub_pcl_.publish(pc2_msg)

        # print('soy feliz')
        # self.state = self.STATE_WAITING_IMAGE_RECT

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
        
        if self.first_sterep_image:
            self.first_sterep_image = False
            self.node.get_logger().info(f'camera info self.left.p {self.left.p}')
            self.node.get_logger().info(f'camera info self.right.p {self.right.p}')
        
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

    def PubMatches(self):
        self.GenericPubImg(self.pub_stereo_matches, self.stereo.DrawMatches())
    
    def PubPointCloud(self):
        header = Header()
        header.frame_id = 'map'
        dtype = PointField.FLOAT32
        point_step = 16
        fields = [
            PointField(name='x', offset=0, datatype=dtype, count=1),
            PointField(name='y', offset=4, datatype=dtype, count=1),
            PointField(name='z', offset=8, datatype=dtype, count=1),
            ]
        pc2_msg = point_cloud2.create_cloud(header, fields, self.stereo.points)

        # pc2_msg = point_cloud2.create_cloud(self.header, self.fields, points)
        self.pub_stereo_pointcloud.publish(pc2_msg)

        # print('soy feliz')
        # self.state = self.STATE_WAITING_IMAGE_RECT

def main(args=None):
    rclpy.init(args=args)
    node = TP3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()