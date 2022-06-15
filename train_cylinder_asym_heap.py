import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import yaml

from prettytable import PrettyTable
from utils.metric_util import per_class_iu, fast_hist_crop, AverageMeter, ProgressMeter
from dataloader.pc_dataset import get_heap_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import polar2cat

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings

warnings.filterwarnings("ignore")

use_wandb = True
if use_wandb:
    wandb.login(key='4d8dd62b978bbed4276d53f03a9e5f4973fc320b')
    run = wandb.init(project="Cylinder3D-Heap-6-classes-with-self-arm", entity="rsl-lidar-seg")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Gradients"])
    total_params = 0
    num_layers = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.size()
        grads = None
        if parameter.grad is not None:
            grads = parameter.grad.size()
        table.add_row([name, params, grads])
        # total_params+=params
        num_layers+=1
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f"Total Number of Layers: {num_layers}")
    return total_params


def main(args):

    # run = wandb.init()
    # artifact = run.use_artifact('rsl-lidar-seg/Cylinder3D-Heap/model:v299', type='model')
    # artifact_dir = artifact.download()
    USE_PREDICTION_THRESHOLD = False
    PREDICTION_THRESHOLD = 0.7

    pytorch_device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_heap_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    with open(dataset_config["label_mapping"], 'r') as stream:
        heap_yaml = yaml.safe_load(stream)
    inv_learning_map = heap_yaml['learning_map_inv']
    color_map = heap_yaml['color_map']

    my_model = model_builder.build(model_config)
    if use_wandb:
        run.watch(my_model)
    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
    if os.path.exists(model_load_path):
        model_dict = torch.load(model_load_path)
        my_model.load_state_dict(state_dict=model_dict['model_state_dict'], strict=True)
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])

    # count_parameters(my_model)
    # my_model.to(pytorch_device)
    # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        # time.sleep(1)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        totaltime = AverageMeter('Total Time', ':6.3f')
        dataconversion_time = AverageMeter('Data conversion Time', ':6.3f')
        forwardtime = AverageMeter('Forward Time', ':6.3f')
        losstime = AverageMeter('Loss Time', ':6.3f')
        backward_time = AverageMeter('Backward Time', ':6.3f')
        optimizer_time = AverageMeter('Optimizer Time', ':6.3f')
        rest_time = AverageMeter('Rest Time', ':6.3f')
        progress = ProgressMeter(
            len(train_dataset_loader),
            [batch_time, data_time, dataconversion_time, forwardtime, losstime, backward_time, optimizer_time, rest_time],
            prefix="Epoch: [{}]".format(epoch))
        print("\n Epoch: ", epoch)
        # torch.cuda.synchronize()
        # end = time.time()
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            # torch.cuda.synchronize()
            # data_time.update(time.time() - end)
            # if global_iter % check_iter == 0 and epoch >= 0:
            if i_iter == 0:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                            val_dataset_loader):
                        print(i_iter_val)
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)

                        if USE_PREDICTION_THRESHOLD:
                            thresholded_labels = np.zeros((predict_labels.shape[0], predict_labels.shape[2],
                                                           predict_labels.shape[3], predict_labels.shape[4]))
                            predict_labels = torch.nn.functional.softmax(predict_labels).cpu().detach().numpy()

                            # Threshold predictions that fall below confidence threshold
                            counter = 0
                            for count, i_val_grid in enumerate(val_grid):
                                for indices in i_val_grid:
                                    max_ind = np.argmax(predict_labels[count, :, indices[0], indices[1], indices[2]])
                                    if predict_labels[count, max_ind, indices[0], indices[1], indices[2]] > PREDICTION_THRESHOLD:
                                        thresholded_labels[count, indices[0], indices[1], indices[2]] = max_ind
                                    else:
                                        counter += 1
                            print("Number of rejected predictions: ", counter)
                            predict_labels = thresholded_labels.astype('int64')
                        else:
                            predict_labels = torch.argmax(predict_labels, dim=1)
                            predict_labels = predict_labels.cpu().detach().numpy()

                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())




                        if use_wandb and i_iter_val % 350 == 0:
                            inv_labels = np.vectorize(inv_learning_map.__getitem__)(predict_labels[count,
                                                                                                   val_grid[count][:,
                                                                                                   0],
                                                                                                   val_grid[count][:,
                                                                                                   1],
                                                                                                   val_grid[count][:,
                                                                                                   2]])
                            points_rgb = np.zeros((inv_labels.shape[0], 6))
                            for label_index, label in enumerate(inv_labels):
                                points_rgb[label_index, 0:3] = polar2cat(val_pt_fea[0][label_index][3:6])
                                color = color_map[label]
                                points_rgb[label_index, 3] = color[0]
                                points_rgb[label_index, 4] = color[1]
                                points_rgb[label_index, 5] = color[2]
                            wandb.log(
                                {
                                    "3d point cloud": wandb.Object3D(
                                    {
                                        "type": "lidar/beta",
                                        "points": points_rgb,
                                    }
                                    )
                                })
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                    if use_wandb:
                        run.log({class_name + 'IoU': class_iou * 100})
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    model_dict = {'epoch': epoch,
                                  'model_state_dict': my_model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'loss': loss}
                    # torch.save(my_model.state_dict(), model_save_path)
                    torch.save(model_dict, model_save_path)
                    if use_wandb:
                        artifact = wandb.Artifact('model', type='model')
                        artifact.add_file(model_save_path)
                        run.log_artifact(artifact)


                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                if use_wandb:
                    run.log({'Validation mean IoU': val_miou})
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))
                if use_wandb:
                    run.log({'Validation loss': np.mean(val_loss_list)})
            # torch.cuda.synchronize()
            # t0 = time.time()
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            # torch.cuda.synchronize()
            # t1 = time.time()


            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            # torch.cuda.synchronize()
            # t2 = time.time()
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)
            # torch.cuda.synchronize()
            # t3 = time.time()
            loss.backward()
            # torch.cuda.synchronize()
            # t4 = time.time()
            optimizer.step()
            # torch.cuda.synchronize()
            # t5 = time.time()
            loss_list.append(loss.item())
            if global_iter % 10 == 0 and use_wandb:
                if len(loss_list) > 0:
                    run.log({'Training loss': np.mean(loss_list)})

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
            # torch.cuda.synchronize()
            # t6 = time.time()
            # torch.cuda.synchronize()
            # batch_time.update(time.time() - end)
            # torch.cuda.synchronize()
            # end = time.time()

            # dataloading_time = t1 -t0
            # forward_time = t2 - t1
            # loss_time = t3 -t2
            # backward_time = t4 -t3
            # optimizer_time = t5 -t4
            # rest_time = t6 - t5
            # dataconversion_time.update(t1 - t0)
            # forwardtime.update(t2 - t1)
            # losstime.update(t3 - t2)
            # backward_time.update(t4 - t3)
            # optimizer_time.update(t5 - t4)
            # rest_time.update(t6 - t5)
            # if global_iter % 10 == 0:
            #     progress.display(global_iter)
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/heap.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
