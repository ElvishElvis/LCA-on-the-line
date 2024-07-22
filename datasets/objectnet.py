import os
import json
import numpy as np
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision.transforms import Compose


OBJECTNET_DIR = "/data3/objectnet"


def get_metadata(root=OBJECTNET_DIR):
    metadata = Path(root) / 'objectnet-1.0' / 'mappings'

    with open(metadata / 'folder_to_objectnet_label.json','r') as f:
        folder_map = json.load(f)
        folder_map = {v: k for k, v in folder_map.items()}
    with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
        objectnet_map = json.load(f)

    with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
        pytorch_map = json.load(f)
        pytorch_map = {v: k for k, v in pytorch_map.items()}
        

    with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
        imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

    folder_to_ids = {}
    for objectnet_name, imagenet_names in objectnet_map.items():
        imagenet_names = imagenet_names.split('; ')
        imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
        folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

    classname_map = {v: k for k, v in folder_map.items()}
    return folder_to_ids, classname_map


def crop(img, border=2):
    width, height = img.size
    cropArea = (border, border, width - border, height - border)
    img = img.crop(cropArea)
    return img


class ObjectNetDataset(datasets.ImageFolder):

    def __init__(self, label_map, path, transform):
        self.label_map = label_map
        super().__init__(path, transform=transform)
        self.samples = [
            d for d in self.samples
            if os.path.basename(os.path.dirname(d[0])) in self.label_map
        ]
        self.imgs = self.samples
        path_name=np.array(self.samples)[:,0]
        if os.path.isfile('./object_net_cache_results.npy'):
            self.results=np.load('./object_net_cache_results.npy',allow_pickle=True)
            self.label=np.load('./object_net_cache_labels.npy',allow_pickle=True)
        else:
            self.results=[]
            self.label=[]
            for index,path in enumerate(path_name):
                if index%100==0:
                    print(index)
                self.results.append(self.transform(self.loader(path)))
                self.label.append(self.label_map[os.path.basename(os.path.dirname(path))])

            self.results=np.stack(self.results)
            self.label=np.array(self.label)
            np.save('./object_net_cache_results',self.results)
            np.save('./object_net_cache_labels',self.label)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.results[index],self.label[index]



def build_objectnet_dataset(preprocess,
                            root=OBJECTNET_DIR,
                            crop_border=2):
    folders_to_ids, classname_map = get_metadata(root)
    subdir = 'objectnet-1.0/images'
    valdir = os.path.join(root, subdir)
    label_map = {name: idx for idx, name in enumerate(sorted(list(folders_to_ids.keys())))}
    if crop_border != None:
        # crop the image to remove the border
        preprocess = Compose([crop, preprocess])
        
    dataset = ObjectNetDataset(label_map, valdir, transform=preprocess)
    rev_class_idx_map = {}
    class_idx_map = {}
    for idx, name in enumerate(sorted(list(folders_to_ids.keys()))):
        rev_class_idx_map[idx] = folders_to_ids[name]
        for imagenet_idx in rev_class_idx_map[idx]:
            class_idx_map[imagenet_idx] = idx
    classnames = [classname_map[c].lower() for c in sorted(list(folders_to_ids.keys()))]
    return dataset, classnames, class_idx_map, rev_class_idx_map


if __name__ == '__main__':
    PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    
    normalize = torchvision.transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    transform = Compose([
        # torchvision.transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    objectnet_dataset, classnames, class_idx_map, rev_class_idx_map = build_objectnet_dataset(transform)
    print('done')