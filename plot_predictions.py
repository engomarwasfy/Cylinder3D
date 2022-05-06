import open3d.ml.torch as ml3d
import argparse
import numpy as np
import json
import cv2
import open3d as o3d
import yaml


def main() -> None:
    """"This script will visualize a point cloud with lables in the format specified by ml3d.
        More specifically it is currently hardcoded to assign the subset of categories that was
        selected by the authors of the SPVNAS paper."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcl-file', type=str, help='pcl file name')
    parser.add_argument('--label-file', type=str, help='label file name')
    parser.add_argument('--label-color-map', type=str, help='label color map file name')
    parser.add_argument('--mask-info-file', type=str, default='', help='mask info file name')
    args, opts = parser.parse_known_args()
    DATASET = 'nuscenes'

    with open(args.label_color_map, "r") as stream:
        try:
            label_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    color_map = label_configs['color_map']

    if DATASET == 'nuscenes':
        pcl = np.fromfile(args.pcl_file, dtype=np.float32).reshape(-1, 5)
    elif DATASET == 'kitti':
        pcl = np.fromfile(args.pcl_file, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(args.label_file, dtype=np.int32)
    if args.mask_info_file != '':
        with open(args.mask_info_file, "r") as read_file:
            mask_info = json.load(read_file)

    labels_as_colors = np.ones((pcl.shape[0], 3))
    for index, label in enumerate(labels):
        color = color_map[label]
        labels_as_colors[index, 0] = color[0] / 255
        labels_as_colors[index, 1] = color[1] / 255
        labels_as_colors[index, 2] = color[2] / 255

    pcd = o3d.geometry.PointCloud()
    full_points_np_ar = np.asarray(pcl)[:, 0:3]
    pcd.points = o3d.utility.Vector3dVector(full_points_np_ar)
    pcd.colors = o3d.utility.Vector3dVector(labels_as_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([42, 39, 71]) / 255.0
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()