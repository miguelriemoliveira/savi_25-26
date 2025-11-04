#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d


view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [10.0, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.87911045824568079, -0.1143707949631662, 0.46269225567601935],
                "lookat": [-14.857198715209961, 8.7558956146240234, 4.6393190026283264],
                "up": [-0.45122740480118839, 0.11291073802962912, 0.88523725316662361],
                "zoom": 0.53999999999999981
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def main():

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------
    filename_rgb1 = '../tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = '../tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Create the rgbd image
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    print(rgbd1)

    filename_rgb2 = '../tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = '../tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)

    # Create the rgbd image
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    print(rgbd2)

    # Show the images using matplotlib
    # plt.subplot(1, 2, 1)
    # plt.title('TUM grayscale image')
    # plt.imshow(rgbd1.color)
    # plt.subplot(1, 2, 2)
    # plt.title('TUM depth image')
    # plt.imshow(rgbd1.depth)
    # plt.show()

    # Obtain the point cloud from the rgbd image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd], zoom=0.35)

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------

    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)

    # # Create entities list of objects to draw
    # entities = [pcd1, axes_mesh]
    # entities = [pcd2, axes_mesh]

    # paint points to get a better visualization
    pcd1.paint_uniform_color([1, 0, 0])  # reg, green, blue
    pcd2.paint_uniform_color([0, 0, 1])
    entities = [pcd1, pcd2, axes_mesh]

    # # Draw the geometries
    o3d.visualization.draw_geometries(entities,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      zoom=view['trajectory'][0]['zoom'],)


if __name__ == '__main__':
    main()
