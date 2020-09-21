from flask import Flask
from flask import request, jsonify, json, redirect
from flask import render_template
import pandas as pd
import numpy as np


from explainx_cf.WebApplication.model import *
from explainx_cf.WebApplication.utils import *
from explainx_cf.WebApplication.individual_explanation import *
from explainx_cf.WebApplication.global_explanations import *
from explainx_cf.WebApplication.queries import *
from explainx_cf.WebApplication.d3_functions import *
from explainx_cf.WebApplication.preprocessing import create_summary_file
from explainx_cf.WebApplication.distance_function import generate_projection_files, reduce_raw_data
from explainx_cf.WebApplication.projection import show_projection2, full_projection
import os
from os import path
from explainx_cf.WebApplication.run import *



# ============= Initialize model =========== #

# --- Setting random seed --- 
np.random.seed(150)

# --- Resets all stored files ---
reset = False

# --- Dataset Selection ---
admissions_dataset = dataset("admissions", [6]) # (Conversion : Good > 0.7 )
diabetes_dataset = dataset("diabetes", [])
fico_dataset = dataset("fico", [0])
heart_dataset = dataset("heart", [1,5,6,8])
delinquency_dataset = dataset("delinquency", [9])
wine_dataset = dataset("wine", [])
paysim_dataset = dataset("paysim", [])

# --- Finance Datasets --- 

dataset_dict = {
    'admissions': admissions_dataset,
    'diabetes': diabetes_dataset,
    'fico': fico_dataset,
    'heart': heart_dataset,
    'delinquency': delinquency_dataset,
    'wine':wine_dataset,
    'paysim':paysim_dataset
}
    

def init_data_model(X,y, model, model_name):
    column_names= X.columns
    X["target"]= y




    ## init_data start
    global data_name, lock, folder_path, data_path, preproc_path, projection_changes_path,reduced_data_path, projection_anchs_path, no_bins, df, model_path, density_fineness, bins_used
    global categorical_cols, monotonicity_arr, feature_selector_input, feature_names, all_data, data, metadata, target, no_samples, no_features, svm_model, bins_centred, X_pos_array, init_vals
    global col_ranges, all_den, all_median, all_mean, high_den, high_median, high_mean, low_den, low_median, low_mean, dict_array, dict_array_orig, percentage_filter_input



    # --- Data initialization ---
    data_name, lock, folder_path, data_path, preproc_path, projection_changes_path, reduced_data_path, projection_anchs_path, no_bins, df, model_path, density_fineness, bins_used = np.zeros(13)
    categorical_cols, monotonicity_arr, feature_selector_input, feature_names, all_data, data, metadata, target, no_samples, no_features, svm_model, bins_centred, X_pos_array, init_vals = np.zeros(14)
    col_ranges, all_den, all_median, all_mean, high_den, high_median, high_mean, low_den, low_median, low_mean, dict_array, dict_array_orig, percentage_filter_input = np.zeros(13)



    dataset= dataset_dict['heart']

    dataset.lock

    # data_name = dataset.name
    data_name= "data"
    lock = dataset.lock

    # --- Path Parameters --- 
    folder_path = "explainx_cf/WebApplication/static/data/" + data_name + '/'
    data_path = folder_path + data_name + ".csv"
    preproc_path = folder_path + data_name + "_preproc.csv"
    projection_changes_path = folder_path + data_name + "_changes_proj.csv"
    projection_anchs_path = folder_path + data_name + "_anchs_proj.csv"
    reduced_data_path = folder_path + data_name + "_raw_proj"

    # print(reduced_data_path)

    no_bins = 21
    bins_used = 20


    df = X


    model_path = "TBD"   # Manual? 

    # --- Advanced Parameters
    density_fineness = 100
    categorical_cols = []  # Categorical columns can be customized # Whether there is order
    # monotonicity_arr = []  # Local test of monotonicity

    feature_names = np.array(df.columns)[:-1]
    all_data = np.array(df.values)

    # --- Split data and target values ---
    data = all_data[:,:-1]
    # data = np.array(data, dtype=float)
    target = all_data[:,-1]

    # --- Filter data by class ---
    high_data = all_data[all_data[:,-1] == 1][:,:-1]
    low_data =  all_data[all_data[:,-1] == 0][:,:-1]

    no_samples, no_features = data.shape




    svm_model= external_models()
    svm_model.set_model(model)
    svm_model.set_model_name(model_name)
    svm_model.set_col_names(column_names)




    bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  # Note: Does not account for categorical features
    all_den, all_median, all_mean = all_kernel_densities(data,feature_names,density_fineness) # Pre-load density distributions
    high_den, high_median, high_mean = all_kernel_densities(high_data,feature_names,density_fineness)
    low_den, low_median, low_mean = all_kernel_densities(low_data,feature_names,density_fineness)

    monotonicity_arr = mono_finder(svm_model, data, col_ranges)


    bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  # Note: Does not account for categorical features
    all_den, all_median, all_mean = all_kernel_densities(data,feature_names,density_fineness) # Pre-load density distributions
    high_den, high_median, high_mean = all_kernel_densities(high_data,feature_names,density_fineness)
    low_den, low_median, low_mean = all_kernel_densities(low_data,feature_names,density_fineness)

    monotonicity_arr = mono_finder(svm_model, data, col_ranges)


    bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  # Note: Does not account for categorical features
    all_den, all_median, all_mean = all_kernel_densities(data,feature_names,density_fineness) # Pre-load density distributions
    high_den, high_median, high_mean = all_kernel_densities(high_data,feature_names,density_fineness)
    low_den, low_median, low_mean = all_kernel_densities(low_data,feature_names,density_fineness)

    monotonicity_arr = mono_finder(svm_model, data, col_ranges)


    # ==== FEATURE SELECTOR ====
    # init_vals = [0,10]
    samples4test = []
    feature_selector_input = []
    for i in range(no_features):
        feature_selector_input.append(prep_feature_selector(data, i, feature_names, col_ranges, no_bins, samples4test))# 0 indexed
    # If no init vals known then leave blank.



    # --- Perform Preprocessing if new data --- 
    if not path.exists(preproc_path): 
        create_summary_file(data, target, svm_model, bins_centred, X_pos_array, init_vals, no_bins, monotonicity_arr, preproc_path, col_ranges, lock)
    elif reset:
            os.remove(preproc_path)
            create_summary_file(data, target, svm_model, bins_centred, X_pos_array, init_vals, no_bins, monotonicity_arr, preproc_path, col_ranges, lock)



    # --- Projection Files ---
    if ((not path.exists(projection_changes_path[:-4]+"_PCA.csv")) or (not path.exists(projection_anchs_path[:-4]+"_PCA.csv"))):
        generate_projection_files(preproc_path, data, target, projection_changes_path, projection_anchs_path) 
    elif reset:
            os.remove(projection_changes_path[:-4]+"_PCA.csv")
            os.remove(projection_anchs_path[:-4]+"_PCA.csv")
            os.remove(projection_changes_path[:-4]+"_TSNE.csv")
            os.remove(projection_anchs_path[:-4]+"_TSNE.csv")
            generate_projection_files(preproc_path, data, target, projection_changes_path, projection_anchs_path) 



    # --- Dimensionality reduction --- 
    if (not path.exists(reduced_data_path+"_TSNE.csv")) or (not path.exists(reduced_data_path+"_PCA.csv")): 
        reduce_raw_data(data, reduced_data_path, "PCA")
        reduce_raw_data(data, reduced_data_path, "TSNE")
    elif reset:
        os.remove(reduced_data_path+"_TSNE.csv")
        os.remove(reduced_data_path+"_PCA.csv")
        reduce_raw_data(data, reduced_data_path, "PCA")
        reduce_raw_data(data, reduced_data_path, "TSNE")



    # --- Metadata ---
    metadata = 	pd.read_csv(preproc_path, index_col=False).values

    # --- Percentage Filter ---
    samples_selected = [x for x in range(100)]

    percentage_filter_input = prep_percentage_filter(metadata, bins_used, samples_selected)

    conf_matrix_input = prep_confusion_matrix(metadata, samples_selected)

    one_compset = prep_complete_data(metadata, data, feature_names, samples_selected ,col_ranges, bins_centred, X_pos_array, 0)


    all_params = {
        'data_name': data_name,
        'lock': lock,
        'folder_path': folder_path,
        'data_path': data_path,
        'preproc_path': preproc_path,
        'projection_changes_path': projection_changes_path,
        'projection_anchs_path': projection_anchs_path,
        'no_bins': no_bins,
        'df': df,
        'model_path': model_path,
        'density_fineness': density_fineness,
        'categorical_cols': categorical_cols,
        'monotonicity_arr': monotonicity_arr,
        'feature_selector_input': feature_selector_input,
        'percentage_filter_input': percentage_filter_input,
        'feature_names': feature_names,
        'all_data': all_data,
        'data': data,
        'metadata':metadata,
        'target': target,
        'no_samples': no_samples,
        'no_features': no_features,
        'svm_model': svm_model,
        'bins_centred': bins_centred,
        'X_pos_array': X_pos_array,
        'init_vals': init_vals,
        'col_ranges': col_ranges,
        'all_den': all_den,
        'all_median': all_median,
        'all_mean': all_mean,
        'dict_array': dict_array,
        'dict_array_orig': dict_array_orig,
        'reduced_data_path':reduced_data_path,
        'bins_used':bins_used
    }
    
    
    return all_params


class explain():
    def __init__(self):
        super(explain, self).__init__()
        self.param= {}
    def counterfactuals(self, X,y, model, model_name="random_forest"):
        # --- Parameter Dictionary ---
        PD = init_data_model(X,y, model, model_name= model_name)


        data_in(PD)
        run_app()
        
    def dataset_heloc(self):
        dataset= pd.read_csv("explainx_cf/heloc_dataset.csv")

        map_riskperformance= {"RiskPerformance": {"Good":1, "Bad":0}}
        dataset.replace(map_riskperformance, inplace=True)
        y= list(dataset["RiskPerformance"])
        X= dataset.drop("RiskPerformance", axis=1)
        return X,y


explainx_cf=explain()
