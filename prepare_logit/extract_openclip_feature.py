'''
Extract logit for OpenCLIP
'''
import pathlib

import sys
sys.path.append('..')
import open_clip
import numpy as np
import torch
import tqdm
import datasets
import copy
import os
import copy
import itertools
import progmet


path_prefix='openCLIP_'
def extract_features_pipeline(dataset, model_file_name,split_name):
    out_dir = pathlib.Path(root+model_file_name)
    out_dir.mkdir(exist_ok=True)

    if os.path.isfile(out_dir / f'{split_name}.pt') and os.path.isfile(out_dir / f'{split_name}_label.pt'):
        print('Loaded pre-computed image feature')
        return
    features, labels = extract_image_features(dataset)
    features_norm= features/features.norm(dim=-1, keepdim=True)
    print(out_dir / f'{split_name}.pt')
    torch.save(features,out_dir / f'{split_name}_unnormed.pt')
    torch.save(features_norm,out_dir / f'{split_name}.pt')
    torch.save(labels,out_dir / f'{split_name}_label.pt')
    print()
    
def extract_image_features(dataset):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
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
            feature_batch = model.encode_image(image_batch)
            feature_batch_=copy.deepcopy(feature_batch.cpu())
            feature_batches.append(feature_batch_.cpu())
            # Important not to keep output of DataLoader. Perform deep copy.
            # https://github.com/pytorch/pytorch/issues/11201#issuecomment-486232056
            label_batch_=copy.deepcopy(label_batch.cpu())
            label_batches.append(label_batch_)
        features = torch.cat(feature_batches,axis=0)
        labels = torch.cat(label_batches,axis=0)
        return features, labels
        
def load_text_feature(full_text_file_path,text_name):
    text_inputs = torch.cat([open_clip.tokenize(f"a photo of a {c}") for c in text_name]).to(device)
    batch_size=256
    iter_=int(np.ceil(len(text_inputs)/batch_size))
    '''
    extract text feature
    '''
    text_features_list=[]
    # Calculate text features
    with torch.no_grad():
        for i in tqdm.tqdm(range(iter_)):
            slice_=slice(int(i*batch_size),int((i+1)*batch_size))
            text_features_ = model.encode_text(text_inputs[slice_,:])
            text_features_list.append(text_features_)
    text_features=torch.concat(text_features_list)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features=text_features.cpu()
    torch.save(text_features,full_text_file_path)
    print(f'save text embedding to {full_text_file_path}')
    return text_features

def zero_shot_exp(model_file_name,text_features):
    print(model_file_name)
    image_file_path=root+model_file_name
    
    image_dataset= datasets.NpzDataset(image_file_path,'test',None)
    image_dataloader = torch.utils.data.DataLoader(
        dataset=image_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=8,
        prefetch_factor=2)
    gt=torch.load(image_file_path+'/test_label.pt')

    '''
    calculate similarity & acc
    '''
    pred_list=[]
    print('Running zero shot experiment...')
    meter = progmet.ProgressMeter('apply', interval_time=5)
    logit_list=[]
    for batch_index, minibatch in enumerate(itertools.islice(meter(image_dataloader), None)):
        inputs, gt_labels = minibatch
        image_features=inputs
        similarity = 100.0 * (image_features.cuda() @ text_features.T.float().cuda())
        logit_list.append(similarity)
        pred=similarity.argmax(-1)
        pred_list.append(pred)
    logit_list_ = torch.concat(logit_list).cpu()
    os.makedirs(root+'llogit',exist_ok=True)
    torch.save(logit_list_, root+'llogit/{}'.format(model_file_name))
    pred_list_ = torch.concat(pred_list).cpu()
    acc=np.mean((pred_list_==gt).numpy())
    print(f'zero shot accuracy for {model_file_name} test split is {acc}')
    

device = torch.device('cuda')
print('preprocess:' +'\n')
root ='./features/'
text_root='./text_features/'
os.makedirs(root, exist_ok = True)
os.makedirs(text_root, exist_ok = True)
'''
imagenet extraction
'''
DATASETS = ['imagenet_r','imagenet','imagenetv2','imagenet_sketch','imagenet_a','objectnet']
model_name_list=open_clip.list_pretrained()[::-1]
for model_name in model_name_list:
    mol_name, mol_setup = model_name
    print(f"\n\n {model_name} \n\n")
    if 'openai' in mol_setup:
        print(f"Skipping {model_name}")
        continue
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(mol_name, pretrained=mol_setup,device=device)
    except Exception as e:
        print(e)
        print(f"Skipping {model_name}")
        continue

    for dataset_name in DATASETS:
        print()
        split_name='test'

        model_file_name=f'{path_prefix}{dataset_name}_{model_name}'
        print(model_file_name)
        if os.path.isdir(root+model_file_name)==True:
            print(f'passing {root+model_file_name}')
            continue
        if 'imagenet' in dataset_name:
            testset, labels, label_map = datasets.build_imagenet_dataset(dataset_name, split_name, preprocess)
            text_name=labels # it's all imageNet textname
            # text_name=np.array(labels)[label_map].tolist()
        else:
            testset, labels, _, label_map = datasets.build_objectnet_dataset(preprocess)
            text_name=labels

        '''
        Extract image embedding
        '''
        os.makedirs(root+model_file_name,exist_ok=True)
        torch.save(label_map, root+model_file_name+ '/label_map.pt')
        print(f"dataset {dataset_name} [split {split_name}] len {len(testset)} with encode model {model_name}")
        extract_features_pipeline(testset, model_file_name,split_name)

        '''
        Extract text embedding
        '''
        full_text_file_path=text_root+model_file_name
        
        if 'imagenet' in dataset_name:
            full_text_file_path=full_text_file_path.replace(dataset_name,'imagenet')
        elif 'objectnet' in dataset_name:
            full_text_file_path=full_text_file_path
        if os.path.isfile(full_text_file_path):
            print('loaded text embedding')
            text_features=torch.load(full_text_file_path)
        else:
            text_features=load_text_feature(full_text_file_path,text_name)

        if 'imagenet' in dataset_name:
            text_features=text_features[label_map,:]
        '''
        Image/Text feature alignment
        '''
        zero_shot_exp(model_file_name,text_features)
