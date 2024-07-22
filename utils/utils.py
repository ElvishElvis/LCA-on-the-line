import os
import numpy as np
import re

def parse_linear_model_result(latent_hierarchy_name_list):


    dataset_name_keys = ['objectnet_','imagenet_sketch_', 'imagenet_a_', 'imagenet_r_', 'imagenetv2_', 'imagenet',]
    def find_key(text):
        for dataset_key in dataset_name_keys:
            pattern = re.compile(dataset_key)
            if pattern.search(text):
                return dataset_key
        return None
    '''
    parse linear model result
    '''
    linear_result_dict_all=[]
    linear_result_path='/home/jiashi/LCA-on-the-line-private/results/linear0702_latent/loggers/'
    linear_result_list=os.listdir(linear_result_path)

    '''
    Contructing {latent_hierarchy_name: linear probe logger using latent_hierarchy_name}
    '''
    matching_dict={}
    for model_name in latent_hierarchy_name_list:
        for logger_name in linear_result_list:
            if model_name in logger_name:
                    matching_dict[model_name] = logger_name
                    break
    logger_list=list(matching_dict.values())
    '''
    Parse result from logger
    '''
    linear_result_dict={dataset:[] for dataset in dataset_name_keys}
    fix_name=lambda name: name.replace("_","")
    for model_name, logger_name in (matching_dict.items()):
        
        model_name_short=model_name.replace('imagenet_','')
        linear_eval_result=parse_logger(os.path.join(linear_result_path,logger_name))
        for index, result in enumerate(linear_eval_result):
            linear_result_dict[dataset_name_keys[index]].append((fix_name(model_name_short),result))

    return linear_result_dict



def parse_logger(file_path,after_interpolation=True):
    def parse_from_pattern(file_path,pattern):
        results = []
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    accuracy = float(match.group(1))
                    results.append(accuracy)
        
        return results
    
    datasets_list=['objectnet_resNet','imagenet_sketch_resNet','imagenet_a_resNet','imagenet_r_resNet','imagenetv2_resNet','imagenet_resNet']
    metric_list=[]
    for datasets in datasets_list:
        if after_interpolation:
            hint_word = rf" {datasets} Test acc = (\d+\.\d+)"
        else:
            hint_word = rf"Interpolation scale 1.0  test accuracy on {datasets}: (\d+\.\d+)"
        pattern = re.compile(hint_word)
        metric_list.append(parse_from_pattern(file_path,pattern))
    metric_list=np.array(metric_list) # (num_dataset, num_checkpoint)
    best_checkpoint_index=np.argmax(np.sum(metric_list[:4], axis=0)) # use checkpoint perform best on o/s/a/r
    best_acc=metric_list[:,best_checkpoint_index]
    return best_acc
    
    