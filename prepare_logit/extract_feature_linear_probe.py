'''
Extract feature for linear probe experiment
'''
import pathlib
import sys
sys.path.append('../')
import numpy as np
import torch
import tqdm
import datasets
import copy
import os
import copy
import itertools
import torch.nn as nn
import torchvision.models as model
import torchvision
from torchvision import transforms

path_prefix = ''
def extract_features_pipeline(dataset, model_file_name, split_name):
    if os.path.isfile(out_dir / f'{split_name}.pt') and os.path.isfile(out_dir / f'{split_name}_label.pt'):
        print('Loaded pre-computed image feature')
        return
    features, labels = extract_features(dataset)
    features_norm = features / features.norm(dim=-1, keepdim=True)
    print(out_dir / f'{split_name}.pt')
    torch.save(features, out_dir / f'{split_name}_unnormed.pt')
    torch.save(features_norm, out_dir / f'{split_name}.pt')
    torch.save(labels, out_dir / f'{split_name}_label.pt')
    print()

def extract_features(dataset):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=False,
        num_workers=8,
        prefetch_factor=2)

    feature_batches = []
    label_batches = []
    
    model.eval()
    with torch.inference_mode():
        for image_batch, label_batch in tqdm.tqdm(loader):
            image_batch = image_batch.to(device)
            feature_batch = model(image_batch)
            feature_batch_ = copy.deepcopy(feature_batch.cpu())
            feature_batches.append(feature_batch_.cpu())
            label_batch_ = copy.deepcopy(label_batch.cpu())
            label_batches.append(label_batch_)
        features = torch.cat(feature_batches, axis=0)
        labels = torch.cat(label_batches, axis=0)
        return features, labels


device = torch.device('cuda')
print('preprocess:' + '\n')
root = '/scratch/jiashi/linear_feature/'
DATASETS = ['imagenet','imagenet_a','imagenet_r','imagenetv2','imagenet_sketch','objectnet']

for model_name in ['vit-b','vit-l','convnext']:
    if model_name=='resNet_18':
        model = torchvision.models.resnet18(weights='DEFAULT').to(device)
        model.fc = torch.nn.Identity() # 512
    elif model_name=='resNet_50':
        model = torchvision.models.resnet50(weights='DEFAULT').to(device)
        model.fc = torch.nn.Identity() # 2048
    elif model_name=='swin':
        model = torchvision.models.swin_b(weights='DEFAULT').to(device)
        model.head=torch.nn.Identity() #1024
    elif model_name=='vit-b':
        model = torchvision.models.vit_b_32(weights='DEFAULT').to(device)
        model.heads.head=torch.nn.Identity() #768
    elif model_name=='vit-l':
        model = torchvision.models.vit_l_32(weights='DEFAULT').to(device)
        model.heads.head=torch.nn.Identity() #1024
    elif model_name=='convnext':
        model = torchvision.models.convnext_tiny(weights='DEFAULT').to(device)
        model.classifier[-1]=torch.nn.Identity() # 768
    
        
    
    model=nn.DataParallel(model)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for dataset_name in DATASETS:
        for split_name in ['train','test']:
            if split_name=='val' or split_name=='train':
                if dataset_name!='imagenet':
                    continue
            print(f'Extracting {dataset_name} {split_name}')
            model_file_name = f'{path_prefix}{dataset_name}_{model_name}'
            if 'imagenet' in dataset_name:
                testset, labels, label_map = datasets.build_imagenet_dataset(dataset_name, split_name, preprocess)
                text_name=np.array(labels)[label_map].tolist()
            else:
                testset, labels, _, label_map = datasets.build_objectnet_dataset(preprocess)
                text_name=labels
            out_dir = pathlib.Path(root + model_file_name)
            out_dir.mkdir(exist_ok=True)
            torch.save(label_map, root+model_file_name + '/label_map.pt')
            print(f"dataset {dataset_name} [split {split_name}] len {len(testset)} with encode model {model_name}")
            extract_features_pipeline(testset, model_file_name,split_name)
    
