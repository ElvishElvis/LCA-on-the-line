import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data


class NpzDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads tensors from an npz file."""

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        self.root=root
        features = torch.load(os.path.join(root, split + '.pt'))
        print(f"[Warning] Converting float16 precision to float32")
        features = features.float()
            
        labels = torch.load(os.path.join(root, split + '_label.pt'))
        self.targets = labels  # For sampling, etc.
        self.tensor_dataset = torch.utils.data.TensorDataset(features, labels)
        self.transform = transform
        self.name=root.split('/')[-1]
        label_map_path=os.path.join(root, 'label_map.pt')
        if os.path.isfile(label_map_path):
            self.label_map = np.array(torch.load(label_map_path))
        else:
            self.label_map=[i for i in range(len(np.unique(labels.numpy())))]
        text_name_path=os.path.join(root, 'text_name.pt')
        if os.path.isfile(text_name_path):
            self.text_name=torch.load(os.path.join(root, 'text_name.pt'))
    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.tensor_dataset)
    


class NpzAttriDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads tensors from an npz file."""

    def __init__(self, root: str, split: str, dataset: str = '', attri_type: str = '', model_name: str = '', float16 = False, transform: Optional[Callable] = None):
        
        dataset_prefix = {"imagenet": "", "imagenet_a": "_imA", "imagenet_r": "_imR", "imagenet_sketch": "_imS", "imagenetv2": "_im2"}
        if float16:
            print(f"[Warning] Using float16 precision")
            features = torch.load(os.path.join(root, 'attributes', f'{attri_type}_{model_name}{dataset_prefix[dataset]}_{split}_sim_ft_f16.pt'))
        else:
            features = torch.load(os.path.join(root, 'attributes', f'{attri_type}_{model_name}{dataset_prefix[dataset]}_{split}_sim_ft.pt'))
        
        labels = torch.load(os.path.join(root, 'features', f'{dataset}_{model_name}', split + '_label.pt'))
        self.label_map = np.array(torch.load(os.path.join(root, 'features', f'{dataset}_{model_name}', 'label_map.pt')))
        self.targets = labels  # For sampling, etc.
        self.tensor_dataset = torch.utils.data.TensorDataset(features, labels)
        self.transform = transform
        self.name=root.split('/')[-1]

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.tensor_dataset)


if __name__ == '__main__':

    DATASET_PATHS = {  # Modify if you saved these datasets elsewhere
    'imagenet': '/home/jiashi/hiercls/resources/features/imagenet_vitb16',
    'imagenetv2': "/home/jiashi/hiercls/resources/features/imagenetv2_vitb16",
    'imagenet_sketch': "/home/jiashi/hiercls/resources/features/imagenet_sketch_vitb16",
    'imagenet_a': "/home/jiashi/hiercls/resources/features/imagenet_a_vitb16",
    'imagenet_r': "/home/jiashi/hiercls/resources/features/imagenet_r_vitb16",
    }

    imagenet_train=NpzDataset(DATASET_PATHS['imagenet'],'train',None)
    imagenet_val=NpzDataset(DATASET_PATHS['imagenet'],'val',None)
    imagenet_test=NpzDataset(DATASET_PATHS['imagenet'],'test',None)

    imagenetv2_test=NpzDataset(DATASET_PATHS['imagenetv2'],'test',None)
    imagenet_sketch_test=NpzDataset(DATASET_PATHS['imagenet_sketch'],'test',None)
    imagenet_a_test=NpzDataset(DATASET_PATHS['imagenet_a'],'test',None)
    imagenet_r_test=NpzDataset(DATASET_PATHS['imagenet_r'],'test',None)
    
    for dataset in [imagenet_train,imagenet_val,imagenet_test,imagenetv2_test,imagenet_sketch_test,imagenet_a_test,imagenet_r_test]:
        print(f"{dataset.name}, {dataset.__len__()}")
    
    DATASET_PATHS = {  # Modify if you saved these datasets elsewhere
    '2018_ina': '/home/jiashi/hiercls/resources/features/2018_ina_vitb16',
    # '2021_ina': "/data3/jiashi/inaturalist2021/",
    }
    ina_2018_train=NpzDataset(DATASET_PATHS['2018_ina'],'train',None)
    ina_2018_test=NpzDataset(DATASET_PATHS['2018_ina'],'test',None)
    for dataset in [ina_2018_train,ina_2018_test]:
        print(f"{dataset.name}, {dataset.__len__()}")
    