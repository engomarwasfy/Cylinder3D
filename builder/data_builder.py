# -*- coding:utf-8 -*-
# author: Xinge
# @file: data_builder.py
import os.path

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32]):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                return_ref=train_ref, label_mapping=label_mapping, nusc=nusc)
    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=False,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=False,
        scale_aug=False,
        transform_aug=False
    )

    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    train_dataset_loader = MultiEpochsDataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     pin_memory=True,
                                                     drop_last=True,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
