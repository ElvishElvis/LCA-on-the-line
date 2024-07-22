import os
import pandas as pd
import torch.nn.functional as F
import pandas as pd
import copy
import torch
import numpy as np
import ml_collections


import hier
from hier import FindLCA
from create_hierarchy import read_txt_hierarchy
from draw_plot import generate_plot,mae
from utils.utils import parse_linear_model_result
from utils.agreement_utils import calculate_agreement,agreement_on_the_line_pred

def calculate_LCA(wordNet_tree,metric_dict,tree_index_gt_label,tree_index_topn_pred,top_n_selector, dataset_name,lca_matrix=None):
    '''
    Calculating LCA distance, which only depend on the class prediction ordering.

    tree_index_gt_label: in tree-index(1372)
    tree_index_topn_pred: in tree-index (1372)
    '''
    def calculate_LCA_helper_matrix(tree_index_gt_label,cur_topn_pred,wordNet_tree,lca_matrix):
        '''
        Compute LCA distance using LCA matrix (like from latent hierarchy)
        '''
        assert len(cur_topn_pred.shape)==2
        lca_matrix=np.max(lca_matrix)-lca_matrix
        # Convert label from tree-index(1372) to class-index(1000)
        tree_to_class_map=dict(zip(wordNet_tree.leaf_subset(),[i for i in range(1000)]))

        class_index_gt_label=np.vectorize(tree_to_class_map.get)(tree_index_gt_label)
        cur_topn_pred_class=np.vectorize(tree_to_class_map.get)(cur_topn_pred)

        lca_distance=lca_matrix[class_index_gt_label[:, np.newaxis],cur_topn_pred_class]
        lca_distance=lca_distance.astype('float')
        '''
        only calculate wrong prediction, filter out correct prediction.
        '''
        correct_pred_index=tree_index_gt_label==cur_topn_pred[:,0]
        lca_distance[correct_pred_index] = np.nan
        result=np.nanmean(lca_distance)
        return result
    def calculate_LCA_helper_wordnet(tree_index_gt_label,cur_topn_pred,wordNet_tree):
        '''
        Compute LCA distance using WordNet (from wordNet_tree.npy)
        '''
        assert len(cur_topn_pred.shape)==2
        find_LCA_ancestor = FindLCA(wordNet_tree)
        '''
        select which measure to use, could be tree depth or information
        '''
        # values_depth=tree.depths()
        # values_depth=-tree.num_leaf_descendants()
        values_entropy=-np.log2(hier.uniform_leaf(wordNet_tree))
        lca_function= values_entropy     

        LCA_ancestor_node=find_LCA_ancestor(tree_index_gt_label.reshape(-1,1),cur_topn_pred)
        lca_gt=lca_function[tree_index_gt_label].reshape(-1,1)
        lca_ancestor_node=lca_function[LCA_ancestor_node]
        lca_pred=lca_function[cur_topn_pred]

        lca_distance=(lca_gt.reshape(-1,1)-lca_ancestor_node).astype(float)
        assert lca_distance.shape[0]*lca_distance.shape[1]==lca_ancestor_node.shape[0]*lca_ancestor_node.shape[1]
        '''
        filter out correct prediction and only calculate LCA on wrong prediction.
        '''
        correct_pred_index=tree_index_gt_label==cur_topn_pred[:,0]
        lca_distance[correct_pred_index] = np.nan
        result=np.nanmean(lca_distance)
        return result
    
    metric_dict['lca_distance']={}
    for top_n in top_n_selector:
        cur_topn_pred=copy.deepcopy(tree_index_topn_pred)[:,:top_n]
        if lca_matrix is not None:
            top_n_lca_dist=calculate_LCA_helper_matrix(tree_index_gt_label,cur_topn_pred,wordNet_tree,lca_matrix)
        else:
            top_n_lca_dist=calculate_LCA_helper_wordnet(tree_index_gt_label,cur_topn_pred,wordNet_tree)
        metric_dict['lca_distance'][top_n]=top_n_lca_dist
        print(f'Top {top_n} {dataset_name} LCA: {top_n_lca_dist} ')
    return metric_dict

def logit_adaption(pred_logit,model_name,dataset_name,label_map):
    """
    Map logit to imageNet class indexs(1000)

    Returns:
        pred_logit in imageNet class index(1000).
    """
    if pred_logit.shape[1]==1000:
        if 'objectnet' in dataset_name or 'objectnet' in model_name:
            num_dat=len(label_map)
            objectnet_logit_list=[]
            for key in label_map:
                value_list=label_map[key]
                # if one object class map to multiple imageNet class, take the max
                max_logit,maindicator_index=torch.max(pred_logit[:,value_list],axis=1) 
                objectnet_logit_list.append(max_logit)
            pred_logit=torch.stack(objectnet_logit_list,axis=1)
        else:    
            pred_logit=pred_logit[:,label_map]
    return pred_logit


def eval_metric(wordNet_tree,gt_label,pred_logit,label_map,model_name,dataset_name,latent_LCA_matrix=None,logit_in_tree_index=False):
    '''
    Assume gt_label in class-index(1000)
    pred_logit could be either in class-index(1000) or tree-index(1372)
    '''
    assert len(gt_label)==len(pred_logit)

    '''
    Top 1 leaf accuracy calculate in class-index(1000)
    '''
    metric_dict={}
    pred_logit= logit_adaption(pred_logit,model_name,dataset_name,label_map)
    if logit_in_tree_index==True: # if prediction tree(1372)
        pred_logit=pred_logit[:,wordNet_tree.leaf_subset()]
    argsort= pred_logit.argsort(axis=1)
    model_predict_class=argsort[:,-1]
    correct_or_false=(model_predict_class==gt_label).numpy()
    leaf_top1_acc=np.mean(correct_or_false)
    leaf_top5_acc=np.sum([np.mean((argsort[:,-i]==gt_label).numpy()) for i in range(5)])
    leaf_top10_acc=np.sum([np.mean((argsort[:,-i]==gt_label).numpy()) for i in range(10)])
    metric_dict['model_predict_class']=" " .join([str(a) for a in model_predict_class.numpy().tolist()])
    metric_dict['leaf_top1_acc']=leaf_top1_acc
    metric_dict['leaf_top5_acc']=leaf_top5_acc
    metric_dict['leaf_top10_acc']=leaf_top10_acc
    print(f'top 1 leaf acc for {model_name} is {leaf_top1_acc}')
    print(f'top 5 leaf acc for {model_name} is {leaf_top5_acc}')
    print(f'top 10 leaf acc for {model_name} is {leaf_top10_acc}')

    '''
    LCA loss 
    Need to evaluate in tree-index(1372), so need to map GT to tree index 
    '''
    # it's objectnet
    if type(label_map)==dict:
        label_map_for_LCA=[item[0] for item in label_map.values()]
    # it's other dataset
    else:
        label_map_for_LCA=label_map

    # Map label_map GT to imageNet class-index(1000) first, then to tree-index(1372)
    class_index_gt_label=gt_label
    tree_index_gt_label=wordNet_tree.leaf_subset()[label_map_for_LCA][class_index_gt_label]

    # topn_pred:(ndata, nclass), top1_pred=topn_pred[:,0]
    topn_pred=(-pred_logit).argsort(axis=1) # sort in descending order
    topn_leaf_pred=topn_pred
    
    tree_index_topn_pred=wordNet_tree.leaf_subset()[label_map_for_LCA][topn_leaf_pred]
    
    metric_dict['lca_distance']={}
    top_n_selector=[1,5] # how many LCA to evaluate
    metric_dict=calculate_LCA(wordNet_tree,metric_dict,tree_index_gt_label,tree_index_topn_pred,top_n_selector, model_name,latent_LCA_matrix)

    return metric_dict

if __name__ == '__main__':    

    import re
    dataset_name_keys = ['objectnet_','imagenet_sketch_', 'imagenet_a_', 'imagenet_r_', 'imagenetv2_', 'imagenet',]
    def find_key(text):
        for dataset_key in dataset_name_keys:
            pattern = re.compile(dataset_key)
            if pattern.search(text):
                return dataset_key
        return None

    datasets_resources={
        'imagenet_sketch_': 'datasets_resources/imagenet_sketch_dataset/',
        'imagenet': 'datasets_resources/imagenet_dataset/',
        'imagenetv2_': 'datasets_resources/imagenetv2_dataset/',
        'imagenet_a_': 'datasets_resources/imagenet_a_dataset/',
        'imagenet_r_': 'datasets_resources/imagenet_r_dataset/',
        'objectnet_': 'datasets_resources/objectnet_dataset/',
    }

    logit_folder_path='/scratch/jiashi/llogit'

    '''
    Choose whether to use WordNet hierarchy or latent hierarchy to compute LCA distance
    '''
    wordNet_tree=np.load('wordNet_tree.npy',allow_pickle=True).item()
    latent_LCA_matrix=None
    use_latent_hierarchy=False
    if use_latent_hierarchy:
        # Load pre-extracted latent hierarchy
        latent_LCA_matrix_list=os.listdir('./datasets_resources/latent_hierarchy')
        latent_LCA_matrix_list=list(filter(lambda a:find_key(a)=='imagenet',latent_LCA_matrix_list)) # only use hierarchy constructed from ImgNet

        latent_LCA_matrix_name="imagenet_vitl14_336.txt" # select one latent hierarchy from latent_LCA_matrix_list
        print(f'Using LCA matrix from {latent_LCA_matrix_name}')
        latent_LCA_matrix=read_txt_hierarchy('./datasets_resources/latent_hierarchy/'+latent_LCA_matrix_name)

    '''
    evaluation start
    '''
    pprefix="ICML_dict"
    if latent_LCA_matrix is not None:
        pprefix=f"Latent_hier_{latent_LCA_matrix_name[:-4]}"
    models_name_list=sorted(os.listdir(logit_folder_path))
    

    metric_save_name=pprefix+'_result_metric_dict'
    if os.path.isfile(metric_save_name):
        result_metric_dict=torch.load(metric_save_name)
        print(f'Loaded pre-extracted {metric_save_name}')
    else:
        result_metric_dict={}
        for dataset_key in dataset_name_keys:
            result_metric_dict[dataset_key]={}
        for model_name in models_name_list:
            dataset_name=find_key(model_name)
            assert dataset_name is not None
            datasets_resources_path=datasets_resources[dataset_name]
            gt_label=torch.load(datasets_resources_path+'test_label.pt')
            label_map=torch.load(datasets_resources_path+'label_map.pt')
            logit_name=os.path.join(logit_folder_path,model_name)
            # we load pre-computed logit(feature before argmax) for all datasets
            pred_logit=torch.load(logit_name)
            print()
            print(model_name)
            metric_dict=eval_metric(wordNet_tree,gt_label,pred_logit,label_map, model_name,dataset_name,latent_LCA_matrix=latent_LCA_matrix)
            result_metric_dict[dataset_name][model_name]=metric_dict
        torch.save(result_metric_dict,metric_save_name)

    coefficient_metric_dict={}
    models_name_list_copy=copy.deepcopy(models_name_list)
    
    '''
    # parse model_groups from ['ALL_','VM_','VLM_'] for plotting
    '''
    for model_groups in ['ALL_']:
        coefficient_metric_dict[model_groups]={}
        models_name_list=copy.deepcopy(models_name_list_copy)
        '''
        filter out VLM/VM
        '''
        VLM_filter_list=['RN','vit','ViT','feature_extractor']
        if 'VM_' in model_groups:
            models_name_list=sorted([item for item in models_name_list if not any(name in item for name in VLM_filter_list)])
        elif 'VLM_' in model_groups:
            models_name_list=sorted( [item for item in models_name_list if any(name in item for name in VLM_filter_list)])
        print(f" \n Runing {len(models_name_list)} model on {model_groups}")

        '''
        Calculate ID ImageNet metric
        '''
        fix_name=lambda name: name.replace("_","")
        sorted_by_name= lambda list_: sorted((list_),key=lambda a:fix_name(a[0]))
        top1_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['leaf_top1_acc']) for model in models_name_list if find_key(model)=='imagenet'])
        top5_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['leaf_top5_acc']) for model in models_name_list if find_key(model)=='imagenet'])
        top10_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['leaf_top10_acc']) for model in models_name_list if find_key(model)=='imagenet'])
        vanilla_LCA_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['lca_distance'][1]) for model in models_name_list if find_key(model)=='imagenet'])
        vanilla5_LCA_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['lca_distance'][5]) for model in models_name_list if find_key(model)=='imagenet'])
        '''
        Parse Linear model result
        '''
        linear_result_dict=parse_linear_model_result([model for model in models_name_list if find_key(model)=='imagenet'])
        '''
        Calculate ID ImageNet agreement metric
        '''
        model_predict_class_imageNet=sorted_by_name([(fix_name(model.replace('imagenet_',"")),result_metric_dict['imagenet'][model]['model_predict_class']) for model in models_name_list if find_key(model)=='imagenet'])
        imageNet_agreement_name=f'{pprefix}_agreement_dict_imageNet_{model_groups}'
        # load pre-caluclated agreement dict
        if os.path.isfile(f'{imageNet_agreement_name}.npy'):
            agreement_dict_imageNet=np.load(f'{imageNet_agreement_name}.npy',allow_pickle=True).item()
        else:
            agreement_dict_imageNet=calculate_agreement(model_predict_class_imageNet)
            np.save(imageNet_agreement_name,agreement_dict_imageNet)
        agreement_dict_imageNet=list(agreement_dict_imageNet.items())
        
        for dataset_name in dataset_name_keys:
            coefficient_metric_dict[model_groups][dataset_name]={}
            cur_models_name_list= [model for model in models_name_list if find_key(model)==dataset_name]
            print(f'\n\n ############################# {dataset_name} ############################# \n\n ')
            '''
            Calculate OOD dataset metric
            '''
            top1_OOD=sorted_by_name([(fix_name(model.replace(dataset_name,"")),result_metric_dict[dataset_name][model]['leaf_top1_acc']) for model in cur_models_name_list])
            top5_OOD=sorted_by_name([(fix_name(model.replace(dataset_name,"")),result_metric_dict[dataset_name][model]['leaf_top5_acc']) for model in cur_models_name_list])
            vanilla_LCA_OOD=sorted_by_name([(fix_name(model.replace(dataset_name,"")),result_metric_dict[dataset_name][model]['lca_distance'][1]) for model in cur_models_name_list])
            '''
            Calculate OOD ImageNet linear metric
            '''
            linear_model_OOD_Top1=sorted_by_name(linear_result_dict[dataset_name])
            '''
            Calculate OOD ImageNet agreement metric
            '''
            model_predict_class_OOD=sorted_by_name([(fix_name(model.replace(dataset_name,"")),result_metric_dict[dataset_name][model]['model_predict_class']) for model in cur_models_name_list])
            OOD_agreement_name=f'{pprefix}_agreement_dict_{dataset_name}_{model_groups}'
            if os.path.isfile(f'{OOD_agreement_name}.npy'):
                agreement_dict_OOD=np.load(f'{OOD_agreement_name}.npy',allow_pickle=True).item()
            else:
                agreement_dict_OOD=calculate_agreement(model_predict_class_OOD)
                np.save(OOD_agreement_name,agreement_dict_OOD)
            agreement_dict_OOD=list(agreement_dict_OOD.items())
            '''
            ###############################################################################################################
            ############################################## Start Comparision ##############################################
            ###############################################################################################################
            '''
            '''
            Compare ID LCA with ID Top 1 to predict OOD top1
            '''
            indicator_axis_name=["ImageNet LCA"]
            target_axis_name=[f'{dataset_name[:-1]} Top1',f'{dataset_name[:-1]} Top5']
            print('\n ########### Compare ID LCA with ID Top 1 to predict OOD top1 ########### \n')
            for indicator_index,indicator_axis in enumerate([vanilla_LCA_imageNet]):  
                for target_index,target_axis in enumerate([top1_OOD,top5_OOD]):
                    baseline_indicator=top1_imageNet
                    baseline_name='ImageNet Top1'
                    '''
                    Plot config
                    '''
                    no_baseline_compare=False
                    save_fig=True
                    plot_name=f"__{indicator_axis_name[indicator_index]}__{target_axis_name[target_index]}__"
                    print(f"\n {plot_name} \n")
                    plot_save_path=f"{pprefix}_{model_groups}{plot_name}"
                    result_dict=generate_plot(indicator_axis,target_axis,baseline_indicator, dataset_name,cur_models_name_list,plot_save_path,baseline_name,model_groups,no_baseline_compare,save_fig)
                    coefficient_metric_dict[model_groups][dataset_name][plot_name]=result_dict
            '''
            Predict OOD agreement with ID agreement
            '''
            indicator_axis_name=["ImageNet agreement"]
            target_axis_name=[f'{dataset_name[:-1]} agreement'] 
            print('\n ########### Predict OOD agreement with ID agreement ########### \n')
            for indicator_index,indicator_axis in enumerate([agreement_dict_imageNet]):  
                for target_index,target_axis in enumerate([agreement_dict_OOD]):
                    baseline_indicator=agreement_dict_imageNet
                    baseline_name='ImageNet agreement'
                    agreement_on_the_line_pred(top1_imageNet,top1_OOD,model_predict_class_imageNet,model_predict_class_OOD)
                    '''
                    Plot config
                    '''
                    no_baseline_compare=True
                    save_fig=False
                    plot_name=f"__{indicator_axis_name[indicator_index]}__{target_axis_name[target_index]}__"
                    print(f"\n {plot_name} \n")
                    plot_save_path=f"{pprefix}_{model_groups}{plot_name}"
                    result_dict=generate_plot(indicator_axis,target_axis,baseline_indicator, dataset_name,cur_models_name_list,plot_save_path,baseline_name,model_groups,no_baseline_compare,save_fig)
                    coefficient_metric_dict[model_groups][dataset_name][plot_name]=result_dict
            '''
            Predict soft label quality with source model LCA that construct latent hierarchy
            '''
            indicator_axis_name=["ImageNet LCA"]
            target_axis_name=[f'{dataset_name[:-1]} linear probe Top1']
            print('\n ########### Predict soft label quality with source models that construct latent hierarchy ########### \n')
            for indicator_index,indicator_axis in enumerate([vanilla_LCA_imageNet]):  
                for target_index,target_axis in enumerate([linear_model_OOD_Top1]):
                    baseline_indicator=top1_imageNet
                    baseline_name='ImageNet Top1'
                    '''
                    Plot config
                    '''
                    no_baseline_compare=True
                    save_fig=False
                    plot_name=f"__{indicator_axis_name[indicator_index]}__{target_axis_name[target_index]}__"
                    print(f"\n {plot_name} \n")
                    plot_save_path=f"{pprefix}_{model_groups}{plot_name}"
                    result_dict=generate_plot(indicator_axis,target_axis,baseline_indicator, dataset_name,cur_models_name_list,plot_save_path,baseline_name,model_groups,no_baseline_compare,save_fig)
                    coefficient_metric_dict[model_groups][dataset_name][plot_name]=result_dict


    torch.save(coefficient_metric_dict,f'./coefficient_metric_dict_{pprefix}')