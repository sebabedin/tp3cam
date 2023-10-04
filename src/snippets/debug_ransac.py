import message_filters
import rclpy
from rclpy.node import Node
# from sensor_msgs.msg import Image, LaserScan, PointCloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class CheStereoCamera(object):

    def __init__(self, left_camera, right_camera):
        self.left_camera = left_camera
        self.right_camera = right_camera

        self.bf_ = cv2.BFMatcher()

    def NewImagesMsg(self, left_img_msg, right_img_msg):
        self.left_camera.NewImageMsg(left_img_msg)
        self.right_camera.NewImageMsg(right_img_msg)

        self.matches = self.bf_.knnMatch(
            self.left_camera.des,
            self.right_camera.des,
            k=2)

class CheCamera(object):

    def __init__(self, camera_info_msg, name=''):
        self.NewCameraInfoMsg(camera_info_msg)
        self.name = name

        self.bridge_ = CvBridge()
        self.orb_ = cv2.ORB_create()
    
    def NewCameraInfoMsg(self, camera_info_msg):
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

class CameraNode(Node):
    
    STATE_WAITING_CAMERA_INFO = 'state_waiting_camera_info'
    STATE_WAITING_IMAGE_RECT = 'state_waiting_image_rect'
    STATE_DO_SOMETHING = 'state_do_something'

    def __init__(self):
        super().__init__('camera_node')

        self.state = self.STATE_WAITING_CAMERA_INFO

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

    def on_camera_info_cb_(self, left_msg, right_msg):
        if self.STATE_WAITING_CAMERA_INFO == self.state:
            self.state = self.STATE_WAITING_IMAGE_RECT

            self.left_camera = CheCamera(left_msg, 'left')
            self.right_camera = CheCamera(right_msg, 'right')
    
    def on_image_rect_cb_(self, left_msg, right_msg):
        if self.STATE_WAITING_IMAGE_RECT == self.state:
            self.state = self.STATE_DO_SOMETHING

            self.stereo_camera = CheStereoCamera(self.left_camera, self.right_camera)
            self.stereo_camera.NewImagesMsg(left_msg, right_msg)

            self.DoSomething()

    def DoSomething(self):
        print(CameraInfo.p)

        kp1_idx = np.array([match[0].queryIdx for match in self.stereo_camera.matches])
        kp2_idx = np.array([match[1].trainIdx for match in self.stereo_camera.matches])

        kp1 = np.array([self.stereo_camera.left_camera.kp[idx].pt for idx in kp1_idx])
        kp2 = np.array([self.stereo_camera.right_camera.kp[idx].pt for idx in kp2_idx])

        # kp1 = self.stereo_camera.left_camera.kp[kp1_idx].pt
        # kp2 = self.stereo_camera.right_camera.kp[kp2_idx].pt

        kp1_3D = np.ones((3, kp1.shape[0]))
        kp2_3D = np.ones((3, kp2.shape[0]))
        kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
        kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()

        T_1w = self.stereo_camera.left_camera.p
        T_2w = self.stereo_camera.right_camera.p

        X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
        X /= X[3]
        X1 = T_1w[:3] @ X
        X2 = T_2w[:3] @ X

        # msg = array_to_pointcloud2(X1)

        self.header = Header()
        self.header.frame_id = 'map'

        points = X1.T

        dtype = PointField.FLOAT32
        point_step = 16
        self.fields = [
            PointField(name='x', offset=0, datatype=dtype, count=1),
            PointField(name='y', offset=4, datatype=dtype, count=1),
            PointField(name='z', offset=8, datatype=dtype, count=1),
            ]
        pc2_msg = point_cloud2.create_cloud(self.header, self.fields, points)

        # pc2_msg = point_cloud2.create_cloud(self.header, self.fields, points)
        self.pub_pcl_.publish(pc2_msg)

        print('soy feliz')
        self.state = self.STATE_WAITING_IMAGE_RECT

        # self.pub_pcl_.publish(msg)

# def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
#     # make it 2d (even if height will be 1)
#     cloud_arr = np.atleast_2d(cloud_arr)

#     cloud_msg = PointCloud2()

#     if stamp is not None:
#         cloud_msg.header.stamp = stamp
#     if frame_id is not None:
#         cloud_msg.header.frame_id = frame_id
    
#     dtype = PointField.FLOAT32
#     point_step = 16
#     fields = [PointField(name='x', offset=0, datatype=dtype, count=1),
#               PointField(name='y', offset=4, datatype=dtype, count=1),
#               PointField(name='z', offset=8, datatype=dtype, count=1),]

#     cloud_msg.height = cloud_arr.shape[0]
#     cloud_msg.width = cloud_arr.shape[1]
#     # cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
#     # cloud_msg.fields = [PointField.FLOAT64]
#     cloud_msg.fields = fields
#     cloud_msg.is_bigendian = False # assumption
#     cloud_msg.point_step = cloud_arr.dtype.itemsize
#     cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
#     cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])
#     cloud_msg.data = cloud_arr.tostring()
#     return cloud_msg 

# type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
#                  (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
#                  (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
# pftype_to_nptype = dict(type_mappings)
# nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# def dtype_to_fields(dtype):
#     '''Convert a numpy record datatype into a list of PointFields.
#     '''
#     fields = []
#     # for field_name in dtype.names:
#     for field_name in [dtype.name]:
#         np_field_type, field_offset = dtype.fields[field_name]
#         pf = PointField()
#         pf.name = field_name
#         if np_field_type.subdtype:
#             item_dtype, shape = np_field_type.subdtype
#             pf.count = np.prod(shape)
#             np_field_type = item_dtype
#         else:
#             pf.count = 1

#         pf.datatype = nptype_to_pftype[np_field_type]
#         pf.offset = field_offset
#         fields.append(pf)
#     return fields

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()
