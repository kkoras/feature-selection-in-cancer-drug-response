# General imports
import multiprocessing
import numpy as np
import pandas as pd
import time
import sys
import dill
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import collections
import os

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.dummy import DummyRegressor

# Add directory to sys.path in order to import custom modules from there.
sys.path.insert(0, "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/Created Modules")
from gdsc_projects_module import DrugGenomeWide, Experiment, ModelingWithFeatureSelection, ModelingResults

# Initialize proper file pathways
drug_annotations = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Drug annotations/Screened_Compounds-March_27th_2018.xlsx"
cell_line_list = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Cell line list (directly from website)/Cell_listThu Aug 16 22_06_49 2018.csv"
gene_expr = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Gene expression/sanger1018_brainarray_ensemblgene_rma-March_2nd_2017.txt"
cnv1 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Copy number variations/cnv_binary_1.csv"
cnv2 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Copy number variations/PANCANCER_Genetic_feature_cna_Mon Aug  6 16_18_51 2018 (kopia).csv"
coding_variants = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Mutation calls/PANCANCER_Genetic_feature_variant_Mon Aug  6 15_45_44 2018.csv"
drug_response = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Sensitivity profiles/v17.3_fitted_dose_response-March_27th_2018.xlsx"

# Call loading function from DrugGenomeWide class
(drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, 
 coding_variants_df, drug_response_df) = DrugGenomeWide.load_data(
    drug_annotations, cell_line_list, gene_expr, 
    cnv1, cnv2, coding_variants, drug_response)

# Load gene mappings
filepath1 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/GDSC - Prediction only with data related to nominal drug targets (minimal approach)/Created data/mapping_from_ensembl_id_to_hgnc_symbol.p"
filepath2 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/GDSC - Prediction only with data related to nominal drug targets (minimal approach)/Created data/mapping_from_hgnc_symbol_to_ensembl_id.p"
DrugGenomeWide.load_mappings(filepath2, filepath1)   # Initialize class variables

# Print shapes of created DataFrames
print("Loading summary:")
print("Drug annotations:", drug_annotations_df.shape)
print("Cell line list", cell_lines_list_df.shape)
print("Gene expression", gene_expression_df.shape)
print("CNV binary:", cnv_binary_df.shape)
print("Coding variants:", coding_variants_df.shape)
print("Drug response:", drug_response_df.shape)


drugs = DrugGenomeWide.create_drugs(drug_annotations_df)



# Hyperparameter space to search on
# Number of trees in random forest
n_estimators = [10, 20, 50, 100, 200, 500]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2, 101, num = 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 101, num = 10)]
# Method of selecting samples for training each tree
criterion = ["mse"]

# Create the param grid
param_grid = {'estimator__n_estimators': n_estimators,
               'estimator__max_features': max_features,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__criterion': criterion}


# Create ModelingWithFeatureSelecction object
rforest_seeds = [22, 37, 44, 55, 78]
split_seeds = [11, 37, 52, 71, 98]

exp = ModelingWithFeatureSelection(name="RandomForest with StandardScaler and feature selection",
              param_grid=param_grid,
              estimator_seeds=rforest_seeds,
              split_seeds=split_seeds,
              n_combinations=30,
              kfolds=3,
              n_combinations_importances=40,
              kfolds_importances=5,
              rforest_jobs=2,
              rforest_refit_jobs=32,
              tuning_jobs=18)


# LOAD CURRENT MODELINGRESULTS OBJECT
with open("../Created data/Results/randomforest_with_standardscaler_and_feature_selection.pkl", "rb") as f:
    exp_results = dill.load(f)
    
feat_numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
               2000, 3000, 4000, 5000, 6000, 8000, 10000]
drug_counter = 0

# LOAD LIST OF DRUG IDS TO COMPUTE ON POKER
with open("rforest_genome_wide_drug_ids_for_poker.pkl", "rb") as f:
    drug_ids_to_compute_on_poker = dill.load(f)

print("List of drug ids to compute on poker:", type(drug_ids_to_compute_on_poker), len(drug_ids_to_compute_on_poker))

for drug_id in drug_ids_to_compute_on_poker:
    # Current drug object
    drug = drugs[drug_id]
    if (drug.name, drug_id) in exp_results.best_k_dict:
        continue
    if drug.name == "Camptothecin":
        continue
    
    # Extract full data
    data = drug.return_full_data(drug_response_df, gene_expression_df, data_combination=["expression"])
    print(drug.name, data.shape)

    # Extract features and labels
    y = data["AUC"]
    X = data.drop(["cell_line_id", "AUC"], axis=1)
    assert X.shape[1] == 17737 and X.shape[0] == y.shape[0]
    

    (test_results_for_splits, dummy_for_splits, importances_for_splits, best_parameters_for_splits, 
                    tuning_seeds_for_splits, best_k) = exp.rf_model_over_data_splits(
        X, y, feat_numbers, log=True, verbose_importances=1)
    
    # Put results into appropriate fields of ModelingResults object
    exp_results.performance_dict[(drug.name, drug_id)] = test_results_for_splits
    exp_results.dummy_performance_dict[(drug.name, drug_id)] = dummy_for_splits
    exp_results.importances_dict[(drug.name, drug_id)] = importances_for_splits
    exp_results.best_params_dict[(drug.name, drug_id)] = best_parameters_for_splits
    exp_results.tuning_seeds_dict[(drug.name, drug_id)] = tuning_seeds_for_splits
    exp_results.best_k_dict[(drug.name, drug_id)] = best_k
    
    # Save the results
    #res_name = exp_results.name.replace(" ", "_").lower() + ".pkl
    with open("../Created data/Results/randomforest_with_standardscaler_and_feature_selection.pkl", "wb") as f:
        dill.dump(exp_results, f)
    
    drug_counter +=1 
    print(drug_counter, "drugs done")
    print()
    print("*" * 50)
    print()

print()
print("SCRIPT FINISHED, ALL DRUGS DONE")
print()
