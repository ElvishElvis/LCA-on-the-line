import re
import os
import copy
import sys
import tqdm
import delu
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import datasets
from scheduler import build_lr_scheduler
from linear_probe_loss import lca_alignment_loss
from main import logit_adaption,eval_metric
from create_hierarchy import read_txt_hierarchy
from linear_probe_utils import load_lca_matrix_from_tree, process_lca_matrix,Logger,createDirIfDoesntExists

class customLinearMLP(nn.Module):

    def __init__(self, in_features, out_classes):
        super(customLinearMLP, self).__init__()

        self.model = nn.Linear(in_features, out_classes)

    def forward(self, inputs):
        out = self.model(inputs)

        return out

def trainBatch(metric_dict,model, train_dataloader, val_dataloader, optimizer, criterion, out_classes, DataOnGPU, max_iters,
            scheduler = None,
            l1_regularize = False, l1_reg_lambda = 2.0,
            device = "cuda:1", num_epochs = 100, model_path = "", logger = None,lambda_=2,interp_model=None):
    model.to(device)

    logger.log(f"l1_regularize = {l1_regularize}")

    best_val_acc = -1
    loss_epoch, acc_epoch = [-1]*2
    loss, acc, val_acc = [-1]*3

    iter_count = 0
    pbar = tqdm.tqdm(range(1, num_epochs + 1))
    
    for epoch in pbar:
        logger.log(f'\n\n\n+++++++++++++++++++++++++++++++++++++++[Epoch-{epoch-1}]+++++++++++++++++++++++++++++++++++++++')
        
        if iter_count > max_iters:
            break
        
        pbar.set_description(f"[Epoch-{epoch-1}] loss = {loss_epoch:.2f}, acc = {acc_epoch:.2f}, val_acc = {val_acc:.2f}")
        
        train_acc = torchmetrics.Accuracy(num_classes = out_classes, average = "weighted", task = "multiclass",top_k=1)
        train_acc.to(device)

        inputs_list = []
        targets_list = []
        preds_list = []
        loss_list = []
        
        model.train()
        pbar_iter = tqdm.tqdm(train_dataloader)
        for idx, (inputs, targets) in enumerate(pbar_iter):
            pbar_iter.set_description(f"loss = {loss:.2f}, acc = {acc:.2f}")
            iter_count += 1
            if iter_count > max_iters:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = computeLoss(criterion, outputs, targets,lambda_)
            if l1_regularize:
                l1_loss = model.input_selector.abs().mean()
                loss = loss + l1_reg_lambda*l1_loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            acc = train_acc(outputs, targets)
            inputs_list.append(inputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
            preds_list.append(outputs.detach().cpu().numpy())
            loss_list.append(loss.detach().cpu().numpy())
            if iter_count > (max_iters*0.0) and (iter_count % 700 == 0 or iter_count == max_iters):
                logger.log(f'\n++++++++++++++++++++[Epoch-{epoch-1}] iter {iter_count} ++++++++++++++++++++')
                _, val_acc = test(model, val_dataloader, criterion, DataOnGPU, [i for i in range(1000)], dataset_name='imagenet',split='val', device = device, logger = logger)
                if val_acc >= best_val_acc:
                    logger.log(f"New best {val_acc} > prev {best_val_acc}")
                    best_val_acc = val_acc

                    saveModel(model, model_path + "_best.pth")
        

        loss_epoch = np.array(loss_list).mean()
        acc_epoch = train_acc.compute()

        if scheduler is not None:
            scheduler.step(acc_epoch)

        if epoch > (num_epochs*0.0) and (epoch % 3 == 0 or epoch == num_epochs):
            logger.log(f"End of epoch {epoch}")

            _, val_acc = test(model, val_dataloader, criterion, DataOnGPU, test_label_map=[i for i in range(1000)], dataset_name='imagenet',split='val',
                                device = device, logger = logger, epoch = epoch)

            if epoch > 5 and val_acc<0.05:
                logger.log(f"Early stop at {val_acc} epoch {epoch}")
                return model, val_acc, val_acc,metric_dict

            if val_acc >= best_val_acc:
                logger.log(f"New best {val_acc} > prev {best_val_acc}")
                best_val_acc = val_acc
                saveModel(model, model_path + "_best.pth")
        if epoch > (num_epochs*0.0) and (epoch % 10 == 0 or epoch == num_epochs):
            metric_dict=global_test(metric_dict,model,logger,interp_model)
        logger.log(f"[Epoch-{epoch}] loss = {loss_epoch:.2f}, acc = {acc_epoch:.2f}, val_acc = {val_acc:.2f}; best_val {best_val_acc:.2f}")
        
    saveModel(model, model_path + "_last.pth")
    
    model = loadModel(model, model_path + "_best.pth")
    
    _, val_acc = test(model, val_dataloader, criterion, DataOnGPU,test_label_map=[i for i in range(1000)], dataset_name='imagenet',split='val',
                        device = device, logger = logger)
    assert val_acc == best_val_acc, "Error! Best model does not match."

    _, best_train_acc =test(model, train_dataloader, criterion, DataOnGPU,test_label_map=[i for i in range(1000)], dataset_name='imagenet',split='train',
            device = device, logger = logger)
    logger.log(f"[Best] Train acc : {best_train_acc}; Val acc = {val_acc}")
    return  model, train_acc, val_acc,metric_dict




def test(model, test_dataloader, criterion, DataOnGPU, test_label_map, dataset_name, split, device = "cuda:1", logger = None, epoch = -1, interp_model=None):

    model.eval()
    '''
    Interpolation with model from CE
    '''
    if interp_model is not None:
        interp_model=torch.load(interp_model)
        state_dict = copy.deepcopy(model.state_dict())
        scale_list=[i*0.1 for i in range(0,11)]
    else:
        scale_list=[0]
    best_logit=None
    best_acc=-1
    for scalee in scale_list:
        if interp_model is not None:
            state_dict_copy=copy.deepcopy(state_dict)
            interp_copy=copy.deepcopy(interp_model)
            
            result_model=copy.deepcopy(state_dict_copy)
            if 'module.model.weight' in interp_copy:
                result_model['module.model.weight']=state_dict_copy['module.model.weight'].cuda()*scalee+interp_copy['module.model.weight'].cuda()*(1-scalee)
                result_model['module.model.bias']=state_dict_copy['module.model.bias'].cuda()*scalee+interp_copy['module.model.bias'].cuda()*(1-scalee)
            else:
                result_model['module.model.weight']=state_dict_copy['module.model.weight'].cuda()*scalee+interp_copy['model.weight'].cuda()*(1-scalee)
                result_model['module.model.bias']=state_dict_copy['module.model.bias'].cuda()*scalee+interp_copy['model.bias'].cuda()*(1-scalee)
                

            model.load_state_dict(result_model)
        with torch.no_grad():
            targets_list = []
            preds_list = []
            features_list = []
            for idx, (inputs, targets) in enumerate(tqdm.tqdm(test_dataloader)):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                outputs=logit_adaption(outputs,dataset_name,dataset_name,test_label_map)

                targets_list.append(targets.detach().cpu().numpy())
                preds_list.append(outputs.detach().cpu().numpy())
        targets_list = np.hstack(targets_list)
        preds_list = np.vstack(preds_list)
        preds_list=torch.tensor(preds_list)

        logger.log("\n\n\n")
        '''
        regular
        '''
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            acc = ( (torch.sigmoid(preds_list).argmax(1)).numpy() == targets_list).mean()
        else:
            acc = (torch.softmax(preds_list, dim = 1).argmax(1).numpy() == targets_list).mean()
        if interp_model is not None:
            logger.log(f"Interpolation scale { scalee}  {split} accuracy on {dataset_name}: {acc}")
        else:
            logger.log(f"{split} accuracy on {dataset_name}: {acc}")


        if (epoch == -1 or (epoch % 5 == 0)) and (epoch % 10 != 0):
            metric_dict=eval_metric(tree,test_dataloader.dataset.targets,torch.tensor(preds_list),test_dataloader.dataset.label_map,dataset_name,dataset_name)
            logger.log(f"Epoch \n")
            logger.log(f"leaf_top1_acc is {metric_dict['leaf_top1_acc']}")
            logger.log(f"leaf_top1_acc is {metric_dict['leaf_top5_acc']}")
            logger.log(f"leaf_top1_acc is {metric_dict['leaf_top10_acc']}")
            logger.log(f"LCA 1 is {metric_dict['lca_distance'][1]}")
        if acc > best_acc:
            best_acc=acc
            best_logit=preds_list
    model.train()

    return best_logit,best_acc

def global_test(metric_dict,model,logger,interp_model=None):
    DataOnGPU=True
    device='cuda:0'
    results_dict = {}
    for test_dataset in [imagenet_test,imagenet_a_test,imagenet_r_test,imagenet_sketch_test,imagenetv2_test,objectnet_test]:

        dataset_name = test_dataset.root
        dataset_name = "_".join(dataset_name.split('/')[-2].split('_')[:-1])
        logger.log(f"\n\n\n *********** {dataset_name} [start] *********** \n\n")
        test_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        test_label_map=test_dataset.label_map
        test_prob_preds, test_acc = test(model, test_dataloader, criterion, DataOnGPU, test_label_map, dataset_name, split='test', device = device, logger = logger,interp_model=interp_model)
        metric_dict=eval_metric(tree,test_dataset.targets,torch.tensor(test_prob_preds),test_label_map,dataset_name,dataset_name)
        logger.log(f"\n [MLP] Test acc = {test_acc} \n")
        logger.log(f"\n [MLP] top-1 LCA = {metric_dict['lca_distance'][1]} \n")
        logger.log(f"\n\n *********** {dataset_name} [end] *********** \n\n\n")
        results_dict[dataset_name] = {'test_acc':test_acc, 'lca_distance':metric_dict['lca_distance']}
    logger.log(f"\n\n\n *********** Consolidated results [start] *********** \n\n")

    for dataset_name in results_dict.keys():
        logger.log(f"\n {dataset_name} Test acc = {results_dict[dataset_name]['test_acc']} \n")
        logger.log(f"\n [MLP] top-1 LCA = {results_dict[dataset_name]['lca_distance'][1]} \n")
    logger.log(f"\n\n\n *********** Consolidated results [end] *********** \n\n")
    return metric_dict

def saveModel(model, path):

    torch.save(model.state_dict(), path)

def loadModel(model, path):

    state_dict = torch.load(path)

    model.load_state_dict(state_dict)

    return model


def computeLoss(criterion, outputs, targets, lambda_ = 2):
    assert lca_criterion is not None
    loss=lca_criterion(outputs, targets, lambda_)
    return loss



def main_runner(metric_dict,model,exp_prefix,interp_model):
                        
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.001, betas = (0.9, 0.999))
    max_iters = 1000000
    scheduler = None
    num_epochs=60
    l1_reg_lambda=2.0
    scheduler = build_lr_scheduler(optimizer,
                        lr_scheduler = "cosine",
                        warmup_iter = 18,
                        max_iter = max_iters,
                        warmup_type = "linear",
                        warmup_lr = 1e-5,
                        verbose = False
                    )
    DataOnGPU=True
    device='cuda:0'
    model.to(device)
    l1_regularize=False


    model_save_name=f'{exp_prefix}'

    file_path = f'./results/linear/files/{model_save_name}'
    createDirIfDoesntExists(file_path)
    os.system(f'cp -rf linear_probe_runner.py {file_path}')
    os.system(f'cp -rf linear_probe_utils.py {file_path}')
    os.system(f'cp -rf linear_probe_loss.py {file_path}')

    logger = Logger(filename = os.path.join(logger_path, f"{model_save_name}_report.txt"))

    logger.log(lca_matrix)
    logger.log(f"\n\n\n *********** Pre-train results *********** \n\n")


    logger.log(f'Running model in {model_save_name}')

    '''
    train the model
    '''
    logger.log(f"\n\n\n *********** Training *********** \n\n")

    model, train_acc, val_acc, metric_dict = trainBatch(metric_dict,model, train_dataloader, val_dataloader, optimizer, criterion, out_classes, DataOnGPU, max_iters,
                            scheduler = scheduler, num_epochs = num_epochs, model_path = model_path+f"{model_save_name}",
                            l1_regularize = l1_regularize, l1_reg_lambda = l1_reg_lambda, device = device,  logger = logger,lambda_=lambda_,interp_model=interp_model)

    logger.log(f"\n [MLP] Train acc = {train_acc}; Val acc = {val_acc} \n")

    '''
    Test the model
    '''
    
    logger.log(f"\n\n\n *********** Post-train results *********** \n\n")
    metric_dict=global_test(metric_dict,model,logger,interp_model)

    logger.log(f"Results in: {os.path.join(logger_path, f'{model_save_name}_report.txt')}")

    logger.log(f"Finished!")

    logger.close()


if __name__ == '__main__':
    dataset_name_keys = ['objectnet_','imagenet_sketch_', 'imagenet_a_', 'imagenet_r_', 'imagenetv2_', 'imagenet',]
    def find_key(text):
        for dataset_key in dataset_name_keys:
            pattern = re.compile(dataset_key)
            if pattern.search(text):
                return dataset_key
        return None
    # Set seed for reproducible results
    delu.improve_reproducibility(20)
    torch.manual_seed(20)
    np.random.seed(20)
    batch_size=1024
    out_classes = 1000
    criterion="CE_alignment"
    for base_model_prefix in ['resNet_18','resNet_50','swin','vit-b','vit-l','convnext']:
        # Note: Please first pre-extract feature for base_model_prefix with prepare_logit/extract_feature_linear_probe.py
        if base_model_prefix=='resNet_18':
            dimension=512
        elif base_model_prefix=='resNet_50':
            dimension=2048
        elif base_model_prefix=='alex':
            dimension=4096
        elif base_model_prefix=='swin':
            dimension=1024
        elif base_model_prefix=='vit-b':
            dimension=768
        elif base_model_prefix=='vit-l':
            dimension=1024
        elif base_model_prefix=='convnext':
            dimension=768
        model_path = './results/linear/models/'
        createDirIfDoesntExists(model_path)
        logger_path = './results/linear/loggers/'
        createDirIfDoesntExists(logger_path)
        metric_dict={}
        # Modify if you saved these datasets elsewhere
        DATASET_PATHS = {  
            'imagenet' :f'/scratch/jiashi/linear_feature/imagenet_{base_model_prefix}/',
            'imagenet_a' :f'/scratch/jiashi/linear_feature/imagenet_a_{base_model_prefix}/',
            'imagenet_r' :f'/scratch/jiashi/linear_feature/imagenet_r_{base_model_prefix}/',
            'imagenet_sketch' :f'/scratch/jiashi/linear_feature/imagenet_sketch_{base_model_prefix}/',
            'imagenetv2' :f'/scratch/jiashi/linear_feature/imagenetv2_{base_model_prefix}/',
            'objectnet' :f'/scratch/jiashi/linear_feature/objectnet_{base_model_prefix}/'
            }
        
        # Prepare DataLoader
        imagenet_train=datasets.NpzDataset(DATASET_PATHS['imagenet'],'train',None)
        imagenet_val=datasets.NpzDataset(DATASET_PATHS['imagenet'],'val',None)
        imagenet_test=datasets.NpzDataset(DATASET_PATHS['imagenet'],'test',None)

        imagenet_a_test=datasets.NpzDataset(DATASET_PATHS['imagenet_a'],'test',None)
        imagenet_r_test=datasets.NpzDataset(DATASET_PATHS['imagenet_r'],'test',None)
        imagenet_sketch_test=datasets.NpzDataset(DATASET_PATHS['imagenet_sketch'],'test',None)
        imagenetv2_test=datasets.NpzDataset(DATASET_PATHS['imagenetv2'],'test',None)
        objectnet_test=datasets.NpzDataset(DATASET_PATHS['objectnet'],'test',None)
        objectnet_test.label_map=objectnet_test.label_map.item()

        train_dataloader = torch.utils.data.DataLoader(
                    dataset=imagenet_train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(
                    dataset=imagenet_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2)

        test_dataloader = torch.utils.data.DataLoader(
                    dataset=imagenet_test,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2)

        ####################################################################################
        '''
        Configable
        '''
        tree_prefix='WordNet'
        models_name_list=sorted(os.listdir('/scratch/jiashi/llogit'))
        latent_hierarchy_list=list(filter(lambda model:find_key(model)=='imagenet',models_name_list))
        
        # WordNet config
        tree=np.load('wordNet_tree.npy',allow_pickle=True).item()
        WordNet_mode='depth'
        double_path=True 

        # Choose hierarchy
        tree_list=['WordNet'] # Using WordNet as soft labels
        # tree_list=latent_hierarchy_list # Using latent hierarchy as soft labels

        for tree_prefix in tree_list:
            temperature_list=[1,25.0] # temperature for scaling LCA distance matrix
            lambda_list=[1,0.03] # lambda for balance CE and soft loss
            alignment_mode_list=[0,2] # whether to apply BCE/CE with soft loss
            for temperature,lambda_,alignment_mode in zip(temperature_list,lambda_list,alignment_mode_list):
                # When not using soft loss, we don't apply interpolation
                if alignment_mode==0:
                    temperature=1
                    lambda_=1
                    interp_model=None
                else:
                    # assume a baseline model trained with temperature=1, lambda_=0 and alignment_mode=0 existed.
                    interp_model=f"/home/jiashi/LCA-on-the-line-private/results/linear/models/{base_model_prefix}_WordNet_depth_lambda_1_temperature1_alignment0_best.pth"

                with torch.no_grad():
                    '''
                    Construct LCA matrix
                    '''
                    if tree_prefix=='WordNet': # tree from WordNet
                        lca_matrix_raw=load_lca_matrix_from_tree(tree,tree_values=WordNet_mode,double_path=double_path)
                    else: # tree from latent hierarchy
                        lca_matrix_raw=read_txt_hierarchy(f"/scratch/jiashi/scratch_file/model_hierarchy/{tree_prefix}.txt")
                    lca_matrix=process_lca_matrix(lca_matrix_raw,tree_prefix,temperature=temperature)
                    '''
                    Construct alignment soft loss
                    '''
                    lca_criterion=lca_alignment_loss(tree,lca_matrix.cuda(),alignment_mode=alignment_mode)
                    '''
                Init exp
                '''
                exp_prefix= "_".join([base_model_prefix,tree_prefix,WordNet_mode])
                exp_prefix +=f"_lambda_{lambda_}_temperature{temperature}_alignment{alignment_mode}"
                # Construct linear model
                model=customLinearMLP(dimension,out_classes)
                model=nn.DataParallel(model)
                main_runner(metric_dict,model,exp_prefix,interp_model)
