import argparse
import numpy as np
import open3d as o3d
import yaml
import torch

from utils.metric_util import f1_score


def main() -> None:
    """"
    This script will visualize a point cloud with labels. The colors for each label are defined by the
    label-color-map.

    Inputs:
    - pcl-file: Path to the point cloud, stored as a .bin file
    -label-file: Path to the label file
    -label-color-map: Path to the label color map, stored as a yaml file (this file must also contain a label mapping)
    -ground-truth-file: Path to the ground truth label file (optional)

    Variables:
    -DATASET: This variable can either be set to 'kitti', 'nuscenes' or 'heap_section', where the name denotes the
              dataset from which the point cloud came from.
    -APPLY_LABEL_MAPPING: This variable should be set to true if you want the label mapping from the label-color-map
              file to be applied.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcl-file', type=str, help='pcl file name')
    parser.add_argument('--label-file', type=str, help='label file name')
    parser.add_argument('--label-color-map', type=str, help='label color map file name')
    parser.add_argument('--ground-truth-file', type=str, help='ground truth file name')
    args, opts = parser.parse_known_args()
    DATASET = 'heap_section'
    APPlY_LABEL_MAPPING = True

    with open(args.label_color_map, "r") as stream:
        try:
            label_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    color_map = label_configs['color_map']
    learning_map = label_configs['learning_map']

    if DATASET == 'nuscenes':
        pcl = np.fromfile(args.pcl_file, dtype=np.float32).reshape(-1, 5)
    elif DATASET == 'kitti':
        pcl = np.fromfile(args.pcl_file, dtype=np.float32).reshape(-1, 4)
    elif DATASET == 'heap_section':
        pcl = np.fromfile(args.pcl_file, dtype=np.float32).reshape(4, -1).T
    labels = np.fromfile(args.label_file, dtype=np.int32)
    if args.ground_truth_file != '':
        ground_truth = np.fromfile(args.ground_truth_file, dtype=np.int32)
        for index, label in enumerate(ground_truth):
            ground_truth[index] = learning_map[label]
            f1_metrics = f1_score(torch.from_numpy(labels), torch.from_numpy(ground_truth))
    labels_as_colors = np.zeros((pcl.shape[0], 3))
    for index, label in enumerate(labels):
        if APPlY_LABEL_MAPPING:
            mapped_label = learning_map[label]
            color = color_map[mapped_label]
            print(color)
        else:
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