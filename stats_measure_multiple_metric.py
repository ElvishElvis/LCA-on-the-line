'''
Statistical Measurements among several groups of metric
'''
import torch
import numpy as np
import pprint
accuracy_dict=torch.load('/home/jiashi/LCA-on-the-line/ICML_dict_result_metric_dict')
accuracy_map=[(name,accuracy_dict['imagenet'][name]['leaf_top1_acc']) for name in accuracy_dict['imagenet'].keys()]
accuracy_map=sorted(accuracy_map,key=lambda a:a[1])
# get model name sorted in ID accuracy
class_order=np.array(accuracy_map)[:,0].tolist()

import os
corr_dir='/scratch/jiashi/scratch_file/customize_LCA_result/customize_corr_save/'
corr_list=os.listdir(corr_dir)
corr_list_order=[[name for name in corr_list].index(name2) for name2 in class_order]
corr_list=np.array(corr_list)[corr_list_order] # now the list is sorted from ID accuracy 
assert np.mean([name in accuracy_dict['imagenet'].keys() for name in corr_list])==1
keys=['objectnet_','imagenet_sketch_', 'imagenet_a_', 'imagenet_r_', 'imagenetv2_', 'imagenet']
final_result_dict=dict(zip(keys,[[]for i in range(len(keys))]))
for dataset_name in keys:
    print(dataset_name)
    ID_corr_list=[]
    LCA_corr_list=[]
    for model_name in corr_list:
        corr_result=torch.load(corr_dir+model_name)['ALL_']
        corr_result_dataset=corr_result[dataset_name]
        # order: pearson_corr, kendall_corr, spearman_corr,r2,mae_R
        top1_imageNet_corr=corr_result_dataset[list(corr_result_dataset.keys())[0]]['top1 imageNet']
        LCA_corr=corr_result_dataset[list(corr_result_dataset.keys())[0]]['ImageNet LCA']
        ID_corr_list.append(top1_imageNet_corr[0])
        LCA_corr_list.append(LCA_corr[0])
    print(np.mean(ID_corr_list),min(ID_corr_list),max(ID_corr_list),np.std(ID_corr_list))
    print(np.mean(LCA_corr_list),min(LCA_corr_list),max(LCA_corr_list),np.std(LCA_corr_list))
    final_result_dict[dataset_name]=(ID_corr_list,LCA_corr_list)
print(1)
        
        




    


