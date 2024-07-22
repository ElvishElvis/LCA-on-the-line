import warnings 
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import hier
# from metric_hierarchy import prepare_weight_LCA_entropy, prepare_weight_LCA_depth_predgt
warnings.filterwarnings("ignore")

def load_wordNet_hierarchy():
    fname = './datasets_resources/imagenet_fiveai.csv'
    with open(fname) as f:
        tree, node_names = hier.make_hierarchy_from_edges(hier.load_edges(f))
    if not os.path.isfile('wordNet_tree.npy'):
        np.save('wordNet_tree_save',tree)
    return tree, node_names

'''
##############################################################################################################
############################# Latent hierarchy ###############################################################
##############################################################################################################
'''
def hierarchical_clustering(normalized_embeddings, max_depth=9):
    '''
    Perform hierarchical clustering

    return cluster_assignments: in each layer, which cluster center does each class assigned to
    '''
    num_classes = normalized_embeddings.shape[0]
    max_depth=int(np.floor(np.log2(num_classes)))
    # Initialize a matrix to store the cluster assignments
    cluster_assignments = np.zeros((max_depth + 1, num_classes), dtype=int)
    for i in range(max_depth + 1):
        num_clusters = 2**i
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(normalized_embeddings)
        cluster_assignments[i] = kmeans.labels_
        
    return cluster_assignments



def create_lca_matrix(cluster_assignments):
    '''
    LCA matrix record the depth in hierarchy of the least common ancestor node between each pair of classes.
    Specifically, lca_matrix[i, j] record the depth of LCA_node(i,j) in hierarchy
    '''
    num_classes = cluster_assignments.shape[1]
    max_depth = cluster_assignments.shape[0] - 1  # Maximum depth will be one less than the number of depths
    
    lca_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):  # print the current progress
        for j in range(i, num_classes):
            # The depth of LCA is the lowest depth at which the classes are in the same cluster
            for depth in range(max_depth, -1, -1):
                if cluster_assignments[depth, i] == cluster_assignments[depth, j]:
                    lca_matrix[i, j] = depth
                    lca_matrix[j, i] = depth
                    break  # stop searching once the LCA is found
    return lca_matrix




def construct_latent_hierachy(source_feature,source_label):
    embeddings_np = source_feature.numpy()
    labels_np = source_label.numpy()
    # Combine the embeddings and labels into a single DataFrame
    data = pd.DataFrame(embeddings_np)
    data['label'] = labels_np
    # Calculate the mean embedding for each class
    mean_embeddings = data.groupby('label').mean()

    # Normalize the mean embeddings, if desired
    normalized_embeddings = normalize(mean_embeddings, axis=1)
    # Execute the clustering and store the results
    cluster_assignments = hierarchical_clustering(normalized_embeddings)
    
    # in the left layer of hierarchy, each class assign to themselves
    cluster_assignments_new=np.concatenate([cluster_assignments,np.array([i for i in range(len(np.unique(gt_label)))]).reshape(1,-1)])
    
    # Create the LCA matrix
    lca_matrix = create_lca_matrix(cluster_assignments_new)

    # Print the LCA matrix
    print(lca_matrix)
    return lca_matrix

def read_txt_hierarchy(path):
    with open(path,'r') as file:
        customize_LCA_matrix=file.readlines()
        customize_LCA_matrix=customize_LCA_matrix[0].split()
        customize_LCA_matrix=np.array(customize_LCA_matrix).astype(int)
        assert len(customize_LCA_matrix)==1000000 # only use latent hierarchy construct from ImageNet
        return customize_LCA_matrix.reshape(1000,1000)

if __name__ == '__main__':
    wordNet_tree,_=load_wordNet_hierarchy()
    datasets_resources={
        'imagenet_sketch_': 'datasets_resources/imagenet_sketch_dataset/',
        'imagenet': 'datasets_resources/imagenet_dataset/',
        'imagenetv2_': 'datasets_resources/imagenetv2_dataset/',
        'imagenet_a_': 'datasets_resources/imagenet_a_dataset/',
        'imagenet_r_': 'datasets_resources/imagenet_r_dataset/',
        'objectnet_': 'datasets_resources/objectnet_dataset/',
    }
    import re
    dataset_name_keys = ['objectnet_','imagenet_sketch_', 'imagenet_a_', 'imagenet_r_', 'imagenetv2_', 'imagenet',]
    def find_key(text):
        for dataset_key in dataset_name_keys:
            pattern = re.compile(dataset_key)
            if pattern.search(text):
                return dataset_key
        return None


    tree=np.load('tree.npy',allow_pickle=True).item()
    logit_folder='/scratch/jiashi/llogit'

    '''
    evaluation 
    '''
    models_name_list=sorted(os.listdir(logit_folder))

    result_metric_dict={}
    for dataset_key in dataset_name_keys:
        result_metric_dict[dataset_key]={}
        
    # for model_name in models_name_list:
    model_name=models_name_list[-1]
    dataset_name=find_key(model_name)
    assert dataset_name is not None
    datasets_resources_path=datasets_resources[dataset_name]
    gt_label=torch.load(datasets_resources_path+'test_label.pt')
    label_map=torch.load(datasets_resources_path+'label_map.pt')
    logit_name=os.path.join(logit_folder,model_name)
    pred_logit=torch.load(logit_name)
    print()
    print(model_name)

    tree=np.load('tree.npy',allow_pickle=True).item()
    '''
    Constract latent hierarchy from current model
    '''
    hierachy=construct_latent_hierachy(pred_logit,gt_label)
    print(hierachy.shape)
    print(hierachy.max())
    os.makedirs('./results/latent_hierarchy', exist_ok=True)
    with open(f'./results/latent_hierarchy/{model_name}.txt','w') as file:
        file.write(" ".join(hierachy.flatten().astype(str).tolist()))
