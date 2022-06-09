import os
import argparse
import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    vis.run() # user changes the view and press "q" to terminate
    param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = [param]
    o3d.io.write_pinhole_camera_trajectory(filename, trajectory)
    vis.destroy_window()

def load_view_point(pcd, filename, capture_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([42, 39, 71]) / 255.0
    opt.point_size = 2
    ctr = vis.get_view_control()
    trajectory = o3d.io.read_pinhole_camera_trajectory(filename)
    params = trajectory.parameters[0]
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(params)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(capture_name)
    vis.destroy_window()


def main() -> None:
    """"This script will visualize a series of point clouds with lables.
        By setting CREATE_NEW_VIEW_POINT to True you can select viewpoint and confirm
        your selection by pressing "q". This viewpoint will be saved as a json file and
        can be used again when you set CREATE_NEW_VIEW_POINT to False."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcl-folder', type=str, help='pcl file name')
    parser.add_argument('--label-folder', type=str, help='label file name')
    parser.add_argument('--label-color-map', type=str, help='label color map file name')
    parser.add_argument('--save-folder', type=str, help='folder to save plots and viewpoint')

    args, opts = parser.parse_known_args()
    DATASET = 'heap_section'
    REDUCED_LABELS = False
    CREATE_NEW_VIEW_POINT = True
    # SAVE_FOLDER = '/home/lterenzi/Pictures/o3d_image_sequences/2021-08-27-11-36-12-009'

    with open(args.label_color_map, "r") as stream:
        try:
            label_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    color_map = label_configs['color_map']
    learning_map = label_configs['learning_map']
    pcl_paths = sorted(os.listdir(args.pcl_folder))
    label_paths = sorted(os.listdir(args.label_folder))


    initial_pcl_path = args.pcl_folder + '/' + pcl_paths[0]
    initial_pcl = np.fromfile(initial_pcl_path, dtype=np.float32).reshape(4, -1).T
    initial_pcd = o3d.geometry.PointCloud()
    initial_full_points_np_ar = np.asarray(initial_pcl)[:, 0:3]
    initial_pcd.points = o3d.utility.Vector3dVector(initial_full_points_np_ar)

    if CREATE_NEW_VIEW_POINT:
        save_view_point(initial_pcd, args.save_folder + '/viewpoint.json')


    for i in tqdm(range(100, len(pcl_paths))):
        pcl_path = args.pcl_folder + '/' + pcl_paths[i]
        label_path = args.label_folder + '/' + label_paths[i]
        if DATASET == 'nuscenes':
            pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
        elif DATASET == 'kitti':
            pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
        elif DATASET == 'heap_section':
            pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(4, -1).T
        labels = np.fromfile(label_path, dtype=np.int32)

        labels_as_colors = np.ones((pcl.shape[0], 3))
        for index, label in enumerate(labels):
            if REDUCED_LABELS:
                mapped_label = learning_map[label]
                color = color_map[mapped_label]
            else:
                color = color_map[label]

            # labels_as_colors[index, 0] = color[0] / 255
            # labels_as_colors[index, 1] = color[1] / 255
            # labels_as_colors[index, 2] = color[2] / 255
            labels_as_colors[index, 0] = 1
            labels_as_colors[index, 1] = 1
            labels_as_colors[index, 2] = 1

        pcd = o3d.geometry.PointCloud()
        full_points_np_ar = np.asarray(pcl)[:, 0:3]
        pcd.points = o3d.utility.Vector3dVector(full_points_np_ar)
        pcd.colors = o3d.utility.Vector3dVector(labels_as_colors)
        capture_name = args.save_folder + '/capture_' + str(i) + '.png'
        load_view_point(pcd, args.save_folder + '/viewpoint.json', capture_name)




if __name__ == '__main__':
    main()