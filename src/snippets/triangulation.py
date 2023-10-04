import cv2
import numpy as np
point_2D = np.array([[17.4485, 709.7993], [17.4382, 709.8409]])
Proj_Matrices = np.array([ [1037.5, -6.9927, -10.0190, -4780.7], [6.9747, 1043.3, -5.8867, -731.9206], [644.7895, 383.4982, -3231.1], [1036.937, -22.8371, -28.3254, -5607.7], [23.0587, 1043.1, 3.1815, -633.4485], [650.4355, 373.6, -15.3504, -3706.5] ])

OutputArray = np.zeros((3,2))
Points_3D = cv2.triangulatePoints(point_2D, Proj_Matrices, OutputArray)