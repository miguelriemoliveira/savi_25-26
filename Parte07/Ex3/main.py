#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d
from random import random


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


class PlaneSegmenter:

    def __init__(self, point_cloud):

        # store the point cloud in a class property
        self.point_cloud = point_cloud

    def segment(self, distance_threshold=0.35, ransac_n=3, num_iterations=100):

        # First detect the plane using ransac
        plane_model, inlier_idxs = self.point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations)
        # print('Number of inliers: ' + str(len(inlier_idxs)))

        # Hessian form of plane representation: ax + by + cz + d = 0
        self.a, self.b, self.c, self.d = plane_model
        # print('Plane coefficients: a=' + str(a) + ', b=' + str(b) + ', c=' + str(c) + ', d=' + str(d))

        # Extract inliers and outliers
        self.point_cloud_inliers = self.point_cloud.select_by_index(inlier_idxs, invert=False)
        self.point_cloud_outliers = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        # Paint inliers and outpliers with colors
        # point_cloud_inliers.paint_uniform_color([0, 1, 0])  # Paint the ground in green
        # point_cloud_outliers.paint_uniform_color([1, 0, 0])  # Paint the objects in red

    def getInliers(self):
        return self.point_cloud_inliers

    def getOutliers(self):
        return self.point_cloud_outliers


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(
        prog='Point cloud processing',
        description='Process point clouds',)

    parser.add_argument('-fn', '--filename', type=str, default='../point_clouds/factory.ply')
    parser.add_argument('-mnp', '--max_number_planes', type=int, default=5)
    # parser.add_argument('-ti', '--target_image', type=str, default='../images/santorini/2.png')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Load the point cloud
    # ------------------------------------
    original_point_cloud = o3d.io.read_point_cloud(args['filename'])
    print(original_point_cloud)

    print('First point:' + str(original_point_cloud.points[0]))
    print('Number of points: ' + str(len(original_point_cloud.points)))

    # axes_mesh = o3d.geometry.create_mesh_coordinate_frame(size=2)
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10)

    # ------------------------------------
    # Iterative procedure which will progressivley segment new planes
    # ------------------------------------
    point_clouds = []
    remaining_point_cloud = deepcopy(original_point_cloud)
    num_iteration = 0

    while True:

        if len(remaining_point_cloud.points) < 10000:  # termination criteria. Stop when there are few points
            print('Terminating due to low number of points')
            break
        elif num_iteration > args['max_number_planes']:
            print('Terminating due to max number of planes reached.')
            break

        print('iteration ' + str(num_iteration) + ' remaining point cloud with ' +
              str(len(remaining_point_cloud.points)))

        seg = PlaneSegmenter(remaining_point_cloud)
        seg.segment()

        inliers = seg.getInliers()
        outliers = seg.getOutliers()

        # Add inliers to new group of points for this plane
        point_clouds.append(inliers)

        # Set outliers as the new remaining point cloud to be segmented in the net iteration
        remaining_point_cloud = outliers

        num_iteration += 1

    # ------------------------------------
    # Add the points from the plane detections with fake colors
    # ------------------------------------
    for point_cloud in point_clouds:
        r, g, b = random(), random(), random()  # randomize a different colors for each group
        point_cloud.paint_uniform_color([r, g, b])  # Paint the ground in green

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------
    # Create entities list of objects to draw
    entities = [original_point_cloud, axes_mesh]

    entities.extend(point_clouds)

    # Draw the geometries
    o3d.visualization.draw_geometries(entities,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      zoom=view['trajectory'][0]['zoom'],)


if __name__ == '__main__':
    main()
