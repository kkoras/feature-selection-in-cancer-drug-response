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
from sklearn.linear_model import Lasso, ElasticNet
from stability_selection import StabilitySelection

# Add directory to sys.path in order to import custom modules from there.
sys.path.insert(0, "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/Created Modules")
from gdsc_projects_module import DrugWithGenesInSamePathways, Experiment, Modeling, ModelingResults

# Display the number of available CPUs
print(multiprocessing.cpu_count())

# LOAD DATA
# Initialize proper file pathways
drug_annotations = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Drug annotations/Screened_Compounds-March_27th_2018.xlsx"
cell_line_list = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Cell line list (directly from website)/Cell_listThu Aug 16 22_06_49 2018.csv"
gene_expr = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Gene expression/sanger1018_brainarray_ensemblgene_rma-March_2nd_2017.txt"
cnv1 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Copy number variations/cnv_binary_1.csv"
cnv2 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Copy number variations/PANCANCER_Genetic_feature_cna_Mon Aug  6 16_18_51 2018 (kopia).csv"
coding_variants = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Mutation calls/PANCANCER_Genetic_feature_variant_Mon Aug  6 15_45_44 2018.csv"
drug_response = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Sensitivity profiles/v17.3_fitted_dose_response-March_27th_2018.xlsx"

# Load dictionary with targets derived from DrugBank
drugbank_targets = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/DrugBank/Created data/drugbank_map_drug_to_targets.p"

# Load dictionary mapping from target genes to genes that occur in same pathways
reactome_pathway_related_genes = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Reactome/Created Reactome Data/map_target_genes_to_genes_involved_in_same_pathways.pkl"

# Filepath to gene expression signatures provided by Merck
signatures = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Created data/Merck Gene Expression Signatures/Data/SignatureScores_GDSC-cellLines_2018-09-27.tsv"

# Call loading function from DrugWithGenesInSamePathways class
(drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, 
 coding_variants_df, drug_response_df, map_drugs_to_drugbank_targets, 
 map_target_genes_to_same_pathways_genes) = DrugWithGenesInSamePathways.load_data(
    drug_annotations, cell_line_list, gene_expr, 
    cnv1, cnv2, coding_variants, drug_response, drugbank_targets, reactome_pathway_related_genes)

# Load gene expression signatures
signatures_df = pd.read_table(signatures)

# Load helper dict for extraction of CNV data
filepath = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Data/Original Data/Genomics of Drug Sensitivity in Cancer/Original GDSC Data/Copy number variations/Created data/"
with open(filepath + "map_cl_id_and_genetic_feature_to_mutation_status.pkl", "rb") as f:
    map_from_cl_id_and_genetic_feature_to_mutation_status = dill.load(f)


# Load gene mappings
filepath1 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/GDSC - Prediction only with data related to nominal drug targets (minimal approach)/Created data/mapping_from_ensembl_id_to_hgnc_symbol.p"
filepath2 = "/home/kkoras/Documents/Projects/Doktorat - Modelling drug efficacy in cancer/Projects/GDSC - Prediction only with data related to nominal drug targets (minimal approach)/Created data/mapping_from_hgnc_symbol_to_ensembl_id.p"
DrugWithGenesInSamePathways.load_mappings(filepath2, filepath1)   # Initialize class variables

# Print shapes of created DataFrames
print("Loading summary:")
print("Drug annotations:", drug_annotations_df.shape)
print("Cell line list", cell_lines_list_df.shape)
print("Gene expression", gene_expression_df.shape)
print("CNV binary:", cnv_binary_df.shape)
print("Coding variants:", coding_variants_df.shape)
print("Drug response:", drug_response_df.shape)
print("DrugBank mapping (number of matched drugs):", len(map_drugs_to_drugbank_targets))
print("Target gene mapping (number of target genes with match):", len(map_target_genes_to_same_pathways_genes))
print("Gene expression signatures:", signatures_df.shape)
print("Number of entries in mapping from cell line and cnv genetic feature to mutation status:",
     len(map_from_cl_id_and_genetic_feature_to_mutation_status))

drugs = DrugWithGenesInSamePathways.create_drugs(drug_annotations_df, map_drugs_to_drugbank_targets, 
                                       map_target_genes_to_same_pathways_genes)
print(len(drugs))

# Set up data classes we want to use
data_types = ["CNV", "mutation", "expression", "tissue", "merck signatures"]

# MODELING

# Hyperparameter space to search on
param_grid = {"estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1., 5., 10., 30., 50., 100.],
              "estimator__l1_ratio": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.]}

# CREATE MODELING AND MODELINGRESULTS OBJECTS
enet_seeds = [22, 37, 44, 55, 78]
split_seeds = [11, 37, 52, 71, 98]

exp = Modeling(name="Pathway genes with signatures - modeling with ENet over few data splits",
              param_grid=param_grid,
              estimator_seeds=enet_seeds,
              split_seeds=split_seeds,
              n_combinations=30,
              kfolds=3,
              max_iter=2000,
              tuning_jobs=4)

# Initialize new ModelingResults object
exp_results = ModelingResults(exp)
print(exp_results.kfolds, exp_results.tuning_jobs, exp_results.scoring, exp_results.max_iter)

# Load previously computed results
# filename = ""
# with open("../Created data/Results/" + filename, "rb") as f:
#     exp_results = dill.load(f)

# ITERATE OVER DRUGS

# Get rid of warnings
import warnings
warnings.filterwarnings("ignore")

drug_counter = 0
log = True   # Controls verbosity during iterating over drugs

# Enter the loop over drugs
for drug_id in drugs:
    drug=drugs[drug_id]
    
    data = drug.return_full_data(drug_response_df,
                     gene_expression_df=gene_expression_df,
                     cnv_binary_df=cnv_binary_df,
                     map_cl_id_and_feature_to_status=map_from_cl_id_and_genetic_feature_to_mutation_status,
                     cell_line_list=cell_lines_list_df,
                     mutation_df=coding_variants_df,
                     merck_signatures_df=signatures_df,
                     data_combination=data_types)
    
    print(data.shape)
    # Delete features (mutations) in which none of cell lines is mutated
    for column in [feat for feat in list(data.columns) if feat[-3:] == "mut" or feat[:3] == "cna"]:
        if data[column].sum() == 0:
            data = data.drop(column, axis=1)
    if data.shape[0] == 0:   # Check if data exists, if not, skip the drug
        continue
    if data.shape[1] < 16:    # That means that data has only features related to tissue
        continue           # so also skip this case
        
    if log:
        print(drug.name, data.shape)
        
    # Extract features and labels
    y = data["AUC"]
    X = data.drop(["cell_line_id", "AUC"], axis=1)
    X.shape[0] == y.shape[0]
        
    # Add data shapes to corresponding dictionary field in ModelingResults
    exp_results.data_shapes[(drug.name, drug_id)] = X.shape
    
    # Compute the results
    (test_results_for_splits, cv_results_for_splits, 
     best_parameters_for_splits, 
     dummy_for_splits, tuning_seeds_for_splits) = exp.enet_model_over_data_splits(X, y, verbose=1, log=True)
    
    # Put results into appropriate fields of ModelingResults object
    exp_results.performance_dict[(drug.name, drug_id)] = test_results_for_splits
    exp_results.dummy_performance_dict[(drug.name, drug_id)] = dummy_for_splits
    exp_results.best_params_dict[(drug.name, drug_id)] = best_parameters_for_splits
    exp_results.tuning_seeds_dict[(drug.name, drug_id)] = tuning_seeds_for_splits
    exp_results.cv_results_dict[(drug.name, drug_id)] = cv_results_for_splits
    
    # Save the results
    #res_name = exp_results.name.replace(" ", "_").lower() + ".pkl
    with open("../Created data/Results/pathway_genes_with_signatures-enet_over_few_data_splits.pkl", "wb") as f:
        dill.dump(exp_results, f)
    
    drug_counter +=1 
    print(drug_counter, "drugs done")
    print()
    print("*" * 50)
    print()
    
print()
print("SCRIPT FINISHED, ALL DRUGS DONE")
print()



