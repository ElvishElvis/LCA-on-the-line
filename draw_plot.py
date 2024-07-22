'''
Plot
'''
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

from utils.agreement_utils import *

def compute_linear_fit(x, y):
    """Returns bias and slope from regression y on x."""
    x = np.array(x)
    y = np.array(y)

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared
def mae(preds, acc):
    return np.mean(np.abs(preds - acc))

def compute_correlation(x, y):
    scaler = MinMaxScaler()
    x_scaled =  MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten()
    y_scaled =  MinMaxScaler().fit_transform(np.array(y).reshape(-1, 1)).flatten()

    # Pearson correlation
    pearson_corr, pearson_p_value = pearsonr(x_scaled, y_scaled)

    # Kendall rank correlation
    kendall_corr, kendall_p_value = kendalltau(x_scaled, y_scaled)

    # Spearman rank-order correlation
    spearman_corr, spearman_p_value = spearmanr(x_scaled, y_scaled)

    # linear fit
    (bias, slope), r2 = compute_linear_fit(x_scaled, y_scaled)
    pred_y = slope * x_scaled + bias
    mae_R=mae(pred_y, y_scaled)


    print('Pearson correlation coefficient:', pearson_corr)
    print('pearson_corr p-value:', pearson_p_value)
    print('Kendall rank correlation coefficient:', kendall_corr)
    print('kendall_corr p-value:', kendall_p_value)
    print('Spearman rank-order correlation coefficient:', spearman_corr)
    print('spearman_corr p-value:', spearman_p_value)

    print(f"(R^2): {r2}")
    print(f"Mean Absolute Error (R^2): {mae_R}")
    print('\n\n')
    return pearson_corr, kendall_corr, spearman_corr,r2,mae_R

def generate_plot(indicator_axis,target_axis,baseline_indicator, dataset_name,models_name_list, plot_save_path, baseline_name, model_group, no_baseline_compare=False,save_fig=False):
    """
    We plot the correlation between x-axis and (y1-axis/y2-axis) respectively.

    Parameters:
        target_axis: x-axis
        baseline_indicator: y1-axis
        indicator_axis: y2-axis

    For example:
        target_axis: ImageNet-A testset Top1 accuracy
        baseline_indicator: ImageNet testset Top1 accuracy
        indicator_axis: ImageNet testset LCA distance

    """
    _,source_axis_name,target_axis_name,_=plot_save_path.split('__')
    source_axis_name=''.join(list(filter( lambda a: "_" not in a, list(source_axis_name))))
    data=sorted_data(indicator_axis,target_axis,baseline_indicator, dataset_name,models_name_list)
    # # Separate the sorted data into x and y lists
    target_axis,indicator_axis, baseline_indicator, models_name_list = zip(*data)

    factor_y=1 
    VLM_filter_list=['RN','vit','ViT','featureextractor']

    is_VLM= [False if not any(name in item for name in VLM_filter_list) else True for item in models_name_list]
    data = {'models_name_list':models_name_list, 'target_axis': np.array(target_axis)*factor_y, 'indicator_axis': indicator_axis, 'baseline_indicator': np.array(baseline_indicator),'is_VLM': is_VLM}
    if save_fig:
        df = pd.DataFrame(data)
        # Convert dataframe to 2 decimal floats
        df = df.round(5)
        # print(df)
        os.makedirs('./results/csv_folder', exist_ok=True)
        df.to_csv(f'./results/csv_folder/{plot_save_path}_{dataset_name}.csv', sep=';')
        # Create the point plot using seaborn
        # sns.set(style="whitegrid")
        sns.set(style="darkgrid")
    
    '''
    append coefficient metric
    '''
    result_dict={}
    if no_baseline_compare==False:
        print(f'\n Baseline: {baseline_name} and {target_axis_name} \n')
        result_dict['baseline_metric']=compute_correlation(baseline_indicator, target_axis)
    print(f'{source_axis_name} and {target_axis_name} \n')
    result_dict[source_axis_name]=compute_correlation(indicator_axis, target_axis)
    if save_fig:
        sns.set(font_scale=2)
        # Set up the figure and axes
        fig, ax1 = plt.subplots(figsize=(10,10))
        ax2 = ax1.twinx()
        # Draw the regression plots
        if no_baseline_compare==False:
            sns.regplot(data=df, y='baseline_indicator', x='target_axis',  color='red', ax=ax1,scatter_kws={"alpha": 0})
            sns.scatterplot(data=df[df['is_VLM']==False],x='target_axis',  y='baseline_indicator', color='magenta', label='ImgNet Top-1 VM',ax=ax1, marker="o",s=200)
            sns.scatterplot(data=df[df['is_VLM']==True],x='target_axis',  y='baseline_indicator', color='red',  label='ImgNet Top-1 VLM',ax=ax1, marker="^",s=200)
        sns.regplot(data=df, y='indicator_axis', x='target_axis', color='green', label='temp', ax=ax2, scatter_kws={"alpha": 0})
        sns.scatterplot(data=df[df['is_VLM']==False], y='indicator_axis', x='target_axis', color='green', label='temp', ax=ax2,marker="o",s=200)
        sns.scatterplot(data=df[df['is_VLM']==True], y='indicator_axis', x='target_axis', color='blue', label='temp', ax=ax2,marker="^",s=200)
        ax2.set_ylim(df['indicator_axis'].min()*0.99,df['indicator_axis'].max()*1.01)

        # Setup axis name
        # ax1.set_xlabel(f"{target_axis_name}",fontsize=30)
        # if no_baseline_compare==False:
        #     ax1.set_ylabel("ImageNet (ID) Top-1 Test Acc",fontsize=30)
        # ax2.set_ylabel(source_axis_name,fontsize=30)
        # plt.title(f"Correlation Plot of ID & LCA metric on {dataset_name[:-1]}",fontsize=22)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.tick_params(axis='x', rotation=45)

        # Set up the legend for ax1
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        labels_1=['VM (ImgNet Top1)','VLM (ImgNet Top1)']
        lines_2, _ = ax2.get_legend_handles_labels()
        lines_2=lines_2[1:]
        labels_2=['VM (ImgNet LCA)','VLM (ImgNet LCA)']
        if no_baseline_compare==False:
            ax1.legend(lines_1, labels_1, loc='upper right',fontsize=25,frameon=True, framealpha=0.8)
        ax2.legend(lines_2, labels_2, loc='center right',fontsize=25,frameon=True, framealpha=0.8)
        
        # Save plot
        os.makedirs('./results/figures_save', exist_ok=True)
        target_axis_name="_".join((target_axis_name.split()))
        plt.savefig(f'./results/figures_save/{source_axis_name}_{target_axis_name}_{model_group}.png',bbox_inches="tight")
        plt.close()
    return result_dict

'''
Output: data sort by OOD top1
'''
def sorted_data(indicator_axis,target_axis,baseline_indicator, dataset_name,models_name_list):
    '''
    each have format [(models_name_list, value)]
    sort by model name
    '''
    fix_name=lambda name: name.replace("_","")
    sorted_by_name= lambda list_: sorted((list_),key=lambda a:fix_name(a[0]))
    indicator_axis=sorted_by_name(indicator_axis)
    target_axis=sorted_by_name(target_axis)
    baseline_indicator=sorted_by_name(baseline_indicator)
    models_name_list=sorted([fix_name(item.replace(dataset_name,"")) for item in models_name_list])

    indicator_axis=[value[1] for value in indicator_axis]
    target_axis=[value[1] for value in target_axis]
    baseline_indicator=[value[1] for value in baseline_indicator]
    '''
    sort data by OOD top1
    '''
    data = sorted(list(zip( target_axis,indicator_axis, baseline_indicator, models_name_list)))

    return data