import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import hier


def createDirIfDoesntExists(path):
    if not os.path.exists(path):
        os.makedirs(path)



"""
Class to record text to file as well as print to cmd.
"""
class Logger():

    def __init__(self, filename) -> None:
    
        self.filename = filename

        self.file = open(self.filename, 'w')

    def log(self, txt):

        print(txt)
        self.file.write(f"\n{txt}\n")

    #Print all of the file contents
    def printAll(self):


        print(f"{self.file}\n\n")

        with open(self.filename, 'r') as f:
            print(f.read())


    def close(self,):

        self.file.close()




def process_lca_matrix(lca_matrix_raw,tree_prefix,temperature=1.0):
    if lca_matrix_raw is None:
        return None
    if tree_prefix!='WordNet':
        result_matrix=np.max(lca_matrix_raw)-lca_matrix_raw
    else:
        result_matrix=lca_matrix_raw
    result_matrix=result_matrix**temperature

    scaler = MinMaxScaler()
    result_matrix=scaler.fit_transform(result_matrix)
    print(result_matrix)
    return torch.from_numpy(result_matrix)
    

def load_lca_matrix_from_tree(tree,tree_values='entropy',double_path=False,num_class=1000):
    assert tree_values in ['entropy','descendant','depth','none']
    if tree_values=='none':
        return None

    tree_index = tree.leaf_subset()
    find_lca = hier.FindLCA(tree)
    # LCA node among each pair of class node, in tree index(1372)
    pairwise_LCA=np.stack([find_lca(i,tree_index) for i in (tree_index)]) 
    assert (pairwise_LCA == pairwise_LCA.T).all()

    # Acquire different scoring for lca matrix
    # Assert values(LCA) < values(class)
    if tree_values == 'entropy':
        node_values = -np.log2(hier.uniform_leaf(tree)) # 1372
    elif tree_values == 'descendant':
        node_values = -tree.num_leaf_descendants()
    elif tree_values == 'depth':
        node_values = tree.depths().astype(np.float32) # 1372
    '''
    Construct LCA matrix:
    (num_class, num_class), the difference in values of gt and LCA node 
    '''
    # standard : values(pred)-values(LCA(pred,gt))
    # double path : (values(pred)-values(LCA(pred,gt))) + (values(gt)-values(LCA(pred,gt)))
    # it only make a difference with depth, as depth of each leaf is difference
    if tree_values == 'depth' and double_path:
        lca_matrix = node_values[tree_index][:, np.newaxis] + node_values[tree_index][:, np.newaxis].T - 2 * node_values[pairwise_LCA] # gt - lca
    else:
        lca_matrix=node_values[tree_index][:, np.newaxis] - node_values[pairwise_LCA]
    return lca_matrix

    


if __name__ == '__main__':   
    tree=np.load('wordNet_tree.npy',allow_pickle=True).item()
    counter=0
    for tree_values in ['entropy','descendant','depth']:
        for double_path in [True,False]:
            for only_keep_topk in [False,True]:
                for topk in [5,100]:
                    # prune some repeated matrix
                    if only_keep_topk==False and topk==5:
                        continue
                    if tree_values in ['entropy','descendant'] and double_path==True:
                        continue
                    print()
                    print(f"{counter} tree_values {tree_values}, double_path {double_path}, only_keep_topk {only_keep_topk}, topk {topk}")
                    load_lca_matrix_from_tree(tree=tree,tree_values=tree_values,double_path=double_path,only_keep_topk=only_keep_topk,topk=topk)
                    counter+=1







    

