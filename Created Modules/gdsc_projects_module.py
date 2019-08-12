#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:41:37 2018

@author: krzysztof

This module contains utilities useful when performing data analysis and drug sensitivity prediction with 
Genomics of Drug Sensitivity in Cancer (GDSC) database. 

Main utilities are Drug classes and Experiment class. All classes beginning with a word "Drug" represent the compound 
coming from GDSC. There is a separate class for every corresponding experiment setup and genomic feature space. All Drug
classes contain methods for extraction and storage of proper input data. Available data types include: gene expression, binary copy number and coding variants, and cell line tissue type. The set of considered genes is represented as "targets"
attribute of Drug classes. 

The Experiment class is dedicated for storage and analysis of results coming from machine learning experiments. Actual 
machine learning is done outside of a class. The Experiment class have methods for storage, analysis and visualisation
of results.

Classes:
    Drug: Basic class representing a compound from GDSC.
    DrugWithDrugBank: Inherits from Drug, accounts for target genes from DrugBank database.
    DrugGenomeWide: Inherits from Drug, designed for using genome-wide gene exression as input data.
    DrugDirectReactome: Inherits from DrugWithDrugBank, uses only input data related to target genes resulting 
            from direct compound-pathway matching from Reactome.
    DrugWithGenesInSamePathways: Inherits from DrugWithDrugBank, uses only input data related to genes that belong in
            the same pathways as target genes.
    Experiment: Designed to store and analyze results coming from machine learning experiments.
                
"""

# Imports
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import collections

# Sklearn imports

from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

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

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, ElasticNet
from stability_selection import StabilitySelection


#################################################################################################################
# Drug class
#################################################################################################################

class Drug(object):
    """Class representing compound from GDSC database. 
    
    This is the most basic, parent class. Different experimental settings will use more specific, 
    children classes. Main function of the class is to create and store input data corresponding to a given
    drug. Five types of data are considered: gene expression, copy number variants, coding variants, gene expression
    signatures, and tumor tissue type. Class instances are initialized with four basic drug properties: ID, name, gene        
    targets and target pathway. Data attributes are stored as pandas DataFrames and are filled using data files 
    from GDSC via corresponding methods.
    
    Attributes:
        gdsc_id (int): ID from GDSC website.
        name (string): Drug name.
        targets (list of strings): Drug's target gene names (HGNC).
        target_pathway (string): Drug's target pathway as provided in GDSC annotations.
        ensembl targets (list of strings): Drug's target genes ensembl IDs. Can have different length 
                    than "targets" because some gene names may not be matched during mapping. Ensembl IDs are
                    needed for gene expression data.
        map_from_hgnc_to_ensembl (dictionary): Dictionary mapping from gene names to ensembl IDs. Created 
                    after calling the "load_mappings" method.
        map_from_ensembl_to_hgnc (dictionary): Dictionary mapping from ensembl IDs to gene names. Created 
                    after calling the "load_mappings" method.
        total_no_samples_screened (int): Number of cell lines screened for that drug. Created after 
                    calling the "extract_drug_response_data" method.
        response_data (DataFrame): DataFrame with screened cell lines for that drug and corresponding AUC or 
                    IC50 values. Created after calling the "extract_drug_response_data" method.
        screened_cell_lines (list of ints): list containing COSMIC IDs representing cell lines screened for 
                    that drug. Created after calling the "extract_screened_cell_lines" method.
        gene_expression_data (DataFrame): DataFrame with gene expression data, considering only
                    target genes. Created after calling the "extract_gene_expression" method
        mutation_data (DataFrame): DataFrame with binary calls for coding variants, considering only
                    target genes. Created after calling the "extract_mutation_data" method.
        cnv_data (DataFrame): DataFrame with binary calls for copu number variants, considering only
                    target genes. Created after calling the "extract_cnv_data" method.
        tissue_data (DataFrame): DataFrame with dummy encoded tumor tissue types in screened cell lines.
                    Dummy encoding results in 13 binary features. Created after calling the 
                    "extract_tissue_data" method.
        full_data (DataFrame): DataFrame with combined data coming from given set of genetic data
                    classes.
    
    Methods:
        Instance methods:
        __init__: Initialize a Drug instance.
        __repr__: Return string representation of an instance, as a command which can be used to create
                    this instance.
        __str__: Return string representation of an instance.
        extract_drug_response_data: Generate a DataFrame with drug-response data.
        extract_screened_cell_lines: Generate a list of COSMIC IDs representing cell lines screened for that
                    drug.
        extract_gene_expression: Generate a DataFrame with gene expression data for drug's screened cell lines
        extract_mutation_data: Generate a DataFrame with binary calls for coding variants.
        extract_cnv_data: Generate a DataFrame with binary calls for copy number variants.
        extract_cnv_data_faster: Generate a DataFrame with binary calls for copy number variants.
        extract_tissue_data: Generate a DataFrame with dummy encoded tissue types.
        extract_merck_signatures_data: Generate a DataFrame with gene expression signatures provided by Merck.
        concatenate_data: Generate a DataFrame containing all desired genetic data classes. Available data
                    classes are: gene expression, coding variants, cnv variants and tissue type.
        create_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data and saves it in corresponding instance's 
                    field.
        return_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data but does not save it.
        
        Class methods:
        load_mappings: Load appropriate dictionaries mapping between ensembl and HGNC.
        
        Static methods:
        create_drugs: Create a dictionary of Drug class objects, each referenced by it's ID 
                    (keys are drug GDSC ID's)
        load_data: Load all needed data files as DataFrames with one function call.
    
    """
    
    # Class variables
    map_from_hgnc_to_ensembl = None
    map_from_ensembl_to_hgnc = None
    
    # Instance methods
    def __init__(self, gdsc_id, name, targets, target_pathway):
        """Intiliaze the class instance with four basic attributes. "Targets" are gene names 
        and get mapped into Ensembl IDs using class mapping variable."""
        self.gdsc_id = gdsc_id
        self.name = name
        self.targets = targets
        self.target_pathway = target_pathway
        self.ensembl_targets = []
        for x in self.targets:
            try:
                self.ensembl_targets.append(self.map_from_hgnc_to_ensembl[x])
            except KeyError:
                pass
        
    
    def extract_drug_response_data(self, sensitivity_profiles_df, metric="AUC"):
        """Generate a DataFrame containing reponses for every cell line screened for that drug.
        
        Arguments:
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC.
            metric (string): Which statistic to use as a response metric (default "AUC").
        
        Returns:
            None
        """
        df = sensitivity_profiles_df[sensitivity_profiles_df.DRUG_ID == self.gdsc_id][
            ["COSMIC_ID", metric]]
        df.columns = ["cell_line_id", metric]   # Insert column with samples ID
        
        self.total_no_samples_screened = df.shape[0]    # Record how many screened cell lines for drug
        self.response_data = df   # Put DataFrame into corresponding field
        
    
    def extract_screened_cell_lines(self, sensitivity_profiles_df):
        """Generate set of cell lines screened for that drug.
        
        Arguments:
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC.
        
        Returns:
            None
        """
        self.screened_cell_lines = list(
            sensitivity_profiles_df[sensitivity_profiles_df.DRUG_ID == self.gdsc_id]["COSMIC_ID"])
    
    def extract_gene_expression(self, gene_expression_df):
        """Generate DataFrame of gene expression data for cell lines screened for this drug, only
        considering drug's target genes.
        
        Arguments:
            gene_expression_df (DataFrame): Original GDSC gene expression DataFrame.
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC.
        
        Returns:
            None
        """
        cell_lines_str = []   # Gene expressesion DF column names are strings
        for x in self.screened_cell_lines:
            cell_lines_str.append(str(x))
        cl_to_extract = []
        for x in cell_lines_str:
            if x in list(gene_expression_df.columns):   
                cl_to_extract.append(x)   # Extract only cell lines contained in gene expression data
        gene_expr = gene_expression_df[
            gene_expression_df.ensembl_gene.isin(self.ensembl_targets)][["ensembl_gene"] + cl_to_extract]
        gene_expr_t = gene_expr.transpose()
        columns = list(gene_expr_t.loc["ensembl_gene"])
        gene_expr_t.columns = columns
        gene_expr_t = gene_expr_t.drop(["ensembl_gene"])
        rows = list(gene_expr_t.index)
        gene_expr_t.insert(0, "cell_line_id", rows)   # Insert columns with cell line IDs
        gene_expr_t.reset_index(drop=True, inplace=True)
        gene_expr_t["cell_line_id"] = pd.to_numeric(gene_expr_t["cell_line_id"])
        self.gene_expression_data = gene_expr_t   # Put DataFrame into corresponding field
    
        
    def extract_mutation_data(self, mutation_df):
        """Generate a DataFrame with binary mutation calls for screened cell lines and target genes.
        
        Arguments:
            mutation_df: DataFrame with original mutation calls from GDSC.
        
        Returns:
            None
        """
        targets = [x + "_mut" for x in self.targets]
        df = mutation_df.copy()[
                mutation_df.cosmic_sample_id.isin(self.screened_cell_lines)]
        df = df[df.genetic_feature.isin(targets)][["cosmic_sample_id", "genetic_feature", "is_mutated"]]
        cosmic_ids = []
        genetic_features = {}
        for feature in df.genetic_feature.unique():
            genetic_features[feature] = []
        for cl_id in df.cosmic_sample_id.unique():
            cosmic_ids.append(cl_id)
            df_cl = df[df.cosmic_sample_id == cl_id]
            for feature in genetic_features:
                mutation_status = df_cl[
                    df_cl.genetic_feature == feature]["is_mutated"].iloc[0]
                genetic_features[feature].append(mutation_status)
        df1 = pd.DataFrame()
        df1.insert(0, "cell_line_id", cosmic_ids)    # Insert column with samples IDs
        for feature in genetic_features:
            df1[feature] = genetic_features[feature]
        self.mutation_data = df1   # Put DataFrame into corresponding field
        
    def extract_cnv_data(self, cnv_binary_df):
        """Generate data containing binary CNV calls for cell lines screened for the drug.
        
        Arguments:
            cnv_binary_df: DataFrame from GDSC download tool with CNV data.
        
        Returns:
            None
        """
        df = cnv_binary_df[cnv_binary_df.cosmic_sample_id.isin(self.screened_cell_lines)]
        features_to_extract = []   # Map drug's targets to CNV features (segments)
        for row in cnv_binary_df.drop_duplicates(subset="genetic_feature").itertuples():
            feature_name = getattr(row, "genetic_feature")
            genes_in_segment = getattr(row, "genes_in_segment").split(",")
            for target in self.targets:
                if target in genes_in_segment:
                    features_to_extract.append(feature_name)   # If target is in any segment, add it to the list
        features_to_extract = list(set(features_to_extract))
        df = df[df.genetic_feature.isin(features_to_extract)]
        cosmic_ids = []
        feature_dict = {}   # Separate lists for every column in final DataFrame
        for feature in df.genetic_feature.unique():
            feature_dict[feature] = []
        for cl_id in df.cosmic_sample_id.unique():
            cosmic_ids.append(cl_id)
            for feature in feature_dict:
                status = df[
                    (df.cosmic_sample_id == cl_id) & (df.genetic_feature == feature)]["is_mutated"].iloc[0]
                feature_dict[feature].append(status)
        new_df = pd.DataFrame()
        for feature in feature_dict:
            new_df[feature] = feature_dict[feature]
        new_df.insert(0, "cell_line_id", cosmic_ids)
        self.cnv_data = new_df
        
        
    def extract_cnv_data_faster(self, cnv_binary_df, map_cl_id_and_feature_to_status):
        """Generate data containing binary CNV calls for cell lines screened for the drug.
        
        Faster implementation than original "extract_cnv_data" by using mapping between genes and 
        genomic segments.
        
        Arguments:
            cnv_binary_df: DataFrame from GDSC download tool with CNV data.
        
        Returns:
            None
        """
        df = cnv_binary_df[cnv_binary_df.cosmic_sample_id.isin(self.screened_cell_lines)]
        features_to_extract = []   # Map drug's targets to CNV features (segments)
        for row in cnv_binary_df.drop_duplicates(subset="genetic_feature").itertuples():
            feature_name = getattr(row, "genetic_feature")
            genes_in_segment = getattr(row, "genes_in_segment").split(",")
            for target in self.targets:
                if target in genes_in_segment:
                    features_to_extract.append(feature_name)   # If target is in any segment, add it to the list
        features_to_extract = list(set(features_to_extract))
        
        df = df[df.genetic_feature.isin(features_to_extract)]
        
        cosmic_ids = []
        feature_dict = {}   # Separate lists for every column in final DataFrame
        for feature in features_to_extract:
            feature_dict[feature] = []
        for cl_id in df.cosmic_sample_id.unique():
            cosmic_ids.append(cl_id)
            for feature in feature_dict:
                status = map_cl_id_and_feature_to_status[(cl_id, feature)]
                feature_dict[feature].append(status)
        new_df = pd.DataFrame()
        for feature in feature_dict:
            new_df[feature] = feature_dict[feature]
        new_df.insert(0, "cell_line_id", cosmic_ids)
        self.cnv_data = new_df
        
        
    def extract_tissue_data(self, cell_line_list):
        """Generate (dummy encoded) data with cell line tissue type.
        
        Arguments:
            cell_line_list (DataFrame): Cell line list from GDSC.
        
        Returns:
            None
        """
        df = cell_line_list[
                cell_line_list["COSMIC_ID"].isin(self.screened_cell_lines)][["COSMIC_ID", "Tissue"]]
        df.rename(columns={"COSMIC_ID": "cell_line_id"}, inplace=True)
        self.tissue_data = pd.get_dummies(df, columns = ["Tissue"])
        
    def extract_merck_signatures_data(self, signatures_df):
        """Generate data with gene expression signature scores for GDSC cell lines, provided by Merck.
        
        Arguments:
            signatures_df (DataFrame): DataFrame with gene signatures for cell lines.
        Returns: 
            None
        """
        # Compute list of screened cell lines as strings with prefix "X" in order to match
        # signatures DataFrame columns
        cell_lines_str = ["X" + str(cl) for cl in self.screened_cell_lines]
        # Compute list of cell lines that are contained in signatures data
        cls_to_extract = [cl for cl in cell_lines_str 
                          if cl in list(signatures_df.columns)]
        # Extract desired subset of signatures data
        signatures_of_interest = signatures_df[cls_to_extract]
        # Transpose the DataFrame
        signatures_t = signatures_of_interest.transpose()
        # Create a list of cell line IDs whose format matches rest of the data
        cl_ids = pd.Series(signatures_t.index).apply(lambda x: int(x[1:]))
        # Insert proper cell line IDs as a new column
        signatures_t.insert(0, "cell_line_id", list(cl_ids))
        # Drop the index and put computed DataFrame in an instance field
        self.merck_signatures = signatures_t.reset_index(drop=True)
    
    def concatenate_data(self, data_combination):
        """Generate data containing chosen combination of genetic data classes.
        
        Arguments:
            data_combination: List of strings containing data classes to be included. Available options are:
                        "mutation", "expression", "CNV", "tissue", "merck signatures".
        
        Returns:
            None
        """
        # Create a list of DataFrames to include
        objects = [self.response_data]
        if "mutation" in data_combination and self.mutation_data.shape[0] > 0:
            objects.append(self.mutation_data)
        if "expression" in data_combination and self.gene_expression_data.shape[0] > 0:
            objects.append(self.gene_expression_data)
        if "CNV" in data_combination and self.cnv_data.shape[0] > 0:
            objects.append(self.cnv_data)
        if "tissue" in data_combination and self.tissue_data.shape[0] > 0:
            objects.append(self.tissue_data)
        if "merck signatures" in data_combination and self.merck_signatures.shape[0] > 0:
            objects.append(self.merck_signatures)
            
        # Find intersection in cell lines for all desirable DataFrames
        cl_intersection = set(list(self.response_data["cell_line_id"]))
        for obj in objects:
            cl_intersection = cl_intersection.intersection(set(list(obj["cell_line_id"])))
        objects_common = []
        for obj in objects:
            objects_common.append(obj[obj["cell_line_id"].isin(cl_intersection)])
            
        # Check if all DataFrames have the same number of samples
        no_samples = objects_common[0].shape[0]
        for obj in objects_common:
            assert obj.shape[0] == no_samples
            obj.sort_values("cell_line_id", inplace=True)
            obj.reset_index(drop=True, inplace=True)
            
        cl_ids = objects_common[0]["cell_line_id"]
        df_concatenated = pd.concat(objects_common, axis=1, ignore_index=False)
        metric = self.response_data.columns[-1]   # Extract the name of metric which was used for sensitivity
        sensitivities = df_concatenated[metric]
        df_concatenated = df_concatenated.drop(["cell_line_id", metric], axis=1)
        df_concatenated.insert(0, "cell_line_id", cl_ids)
        df_concatenated.insert(df_concatenated.shape[1], metric, sensitivities)
        self.full_data = df_concatenated
        
    def create_full_data(self, sensitivity_profiles_df, gene_expression_df=None, cnv_binary_df=None,
                         map_cl_id_and_feature_to_status=None,
                        cell_line_list=None, mutation_df=None, merck_signatures_df=None,
                         data_combination=None, metric="AUC"):
        """Combine extraction methods in one to generate a DataFrame with desired data.
        
        When calling a function, original DataFrames parsed should match strings in 
        data_combination argument. If any of the "_df" arguments is None (default value),
        the corresponding data is not included in the output DataFrame.
        
        Arguments:
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC.
            gene_expression_df (DataFrame): Original GDSC gene expression DataFrame.
            cnv_binary_df (DataFrame): DataFrame from GDSC download tool with CNV data.
            cell_line_list (DataFrame): Cell line list from GDSC.
            mutation_df (DataFrame): DataFrame with original mutation calls from GDSC.
            data_combination (list): list of strings containing data classes to be included. Available 
                        options are: "mutation", "expression", "CNV, "tissue", "merck signatures".
            metric (string): Which statistic to use as a response metric (default "AUC").
        
        Returns:
            DataFrame containing desired data for the drug
        """
        # Call separate methods for distinct data types
        self.extract_screened_cell_lines(sensitivity_profiles_df)
        self.extract_drug_response_data(sensitivity_profiles_df, metric)
        if type(gene_expression_df) == type(pd.DataFrame()):
            self.extract_gene_expression(gene_expression_df)
        if type(cnv_binary_df) == type(pd.DataFrame()):
            self.extract_cnv_data_faster(cnv_binary_df, map_cl_id_and_feature_to_status)
        if type(cell_line_list) == type(pd.DataFrame()):
            self.extract_tissue_data(cell_line_list)
        if type(mutation_df) == type(pd.DataFrame()):
            self.extract_mutation_data(mutation_df)
        if type(merck_signatures_df) == type(pd.DataFrame()):
            self.extract_merck_signatures_data(merck_signatures_df)
            
        self.concatenate_data(data_combination)
        return self.full_data
    
    def return_full_data(self, sensitivity_profiles_df, gene_expression_df=None, cnv_binary_df=None,
                         map_cl_id_and_feature_to_status=None,
                        cell_line_list=None, mutation_df=None, merck_signatures_df=None,
                         data_combination=None, metric="AUC"):
        """Compute full data with desired data classes and return it, but after that delete data from
        instance's data fields in order to save memory.
        
        When calling a function, original DataFrames parsed should match strings in 
        data_combination argument. If any of the "_df" arguments is None (default value),
        the corresponding data is not included in the output DataFrame.
        
        Arguments:
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC.
            gene_expression_df (DataFrame): Original GDSC gene expression DataFrame.
            cnv_binary_df (DataFrame): DataFrame from GDSC download tool with CNV data.
            cell_line_list (DataFrame): Cell line list from GDSC.
            mutation_df (DataFrame): DataFrame with original mutation calls from GDSC.
            data_combination (list): list of strings containing data classes to be included. Available 
                        options are: "mutation", "expression", "CNV, "tissue", "merck signatures".
            metric (string): Which statistic to use as a response metric (default "AUC").
        
        Returns:
            DataFrame containing desired data for the drug
        """
        full_df = self.create_full_data(sensitivity_profiles_df, gene_expression_df, cnv_binary_df,
                                        map_cl_id_and_feature_to_status,
                        cell_line_list, mutation_df, merck_signatures_df,
                         data_combination, metric)
        if type(gene_expression_df) == type(pd.DataFrame()):
            self.gene_expression_data = None
        if type(cnv_binary_df) == type(pd.DataFrame()):
            self.cnv_data = None
        if type(cell_line_list) == type(pd.DataFrame()):
            self.tissue_data = None
        if type(mutation_df) == type(pd.DataFrame()):
            self.mutation_data = None
        if type(merck_signatures_df) == type(pd.DataFrame()):
            self.merck_signatures = None
        self.full_data = None
        return full_df
        
        
        
    def __repr__(self):
        """Return string representation of an object, which can be used to create it."""
        return 'Drug({}, "{}", {}, "{}")'.format(self.gdsc_id, self.name, self.targets, self.target_pathway)
    
    def __str__(self):
        """Return string representation of an object"""
        return "{} -- {}".format(self.name, self.gdsc_id)
    
    
    # Class methods
    @classmethod
    def load_mappings(cls, filepath_hgnc_to_ensembl, filepath_ensembl_to_hgnc):
        """Load dictonaries with gene mappings between HGNC and Ensembl (from pickle files) and assign it 
        to corresponding class variables. Ensembl IDs are needed for gene expression data.
        
        This method should be called on a Drug class before any other actions with the class.
        
        Arguments:
        filepath_hgnc_to_ensembl: file with accurate mapping
        filepath_ensembl_to_hgnc: file with accurate mapping
        
        Returns:
        None
        """
        cls.map_from_hgnc_to_ensembl = pickle.load(open(filepath_hgnc_to_ensembl, "rb"))
        cls.map_from_ensembl_to_hgnc = pickle.load(open(filepath_ensembl_to_hgnc, "rb"))
        
    # Static methods
    @staticmethod
    def create_drugs(drug_annotations_df):
        """Create a dictionary of Drug class objects, each referenced by it's ID (keys are drug GDSC ID's).

        Arguments:
        drug_annotations_df (DataFrame): DataFrame of drug annotations from GDSC website

        Returns:
        Dictionary of Drug objects as values and their ID's as keys
        """
        drugs = {}
        for row in drug_annotations_df.itertuples(index=True, name="Pandas"):
            gdsc_id = getattr(row, "DRUG_ID")
            name = getattr(row, "DRUG_NAME")
            targets = getattr(row, "TARGET").split(", ")
            target_pathway = getattr(row, "TARGET_PATHWAY")

            drugs[gdsc_id] = Drug(gdsc_id, name, targets, target_pathway)
        return drugs
    
    @staticmethod
    def load_data(drug_annotations, cell_line_list, gene_expr, cnv1, cnv2, 
              coding_variants, drug_response):
        """Load all needed files by calling one function and return data as tuple of DataFrames. All 
        argumenst are filepaths to corrresponding files."""
        # Drug annotations
        drug_annotations_df = pd.read_excel(drug_annotations)
        # Cell line annotations
        col_names = ["Name", "COSMIC_ID", "TCGA classification", "Tissue", "Tissue_subtype", "Count"]
        cell_lines_list_df = pd.read_csv(cell_line_list, usecols=[1, 2, 3, 4, 5, 6], header=0, names=col_names)
        # Gene expression
        gene_expression_df = pd.read_table(gene_expr)
        # CNV
        d1 = pd.read_csv(cnv1)
        d2 = pd.read_table(cnv2)
        d2.columns = ["genes_in_segment"]
        def f(s):
            return s.strip(",")
        cnv_binary_df = d1.copy()
        cnv_binary_df["genes_in_segment"] = d2["genes_in_segment"].apply(f)
        # Coding variants
        coding_variants_df = pd.read_csv(coding_variants)
        # Drug-response
        drug_response_df = pd.read_excel(drug_response)
        return (drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, coding_variants_df,
               drug_response_df)

#################################################################################################################
# DrugWithDrugBank class
##################################################################################################################

class DrugWithDrugBank(Drug):
    """Class representing drug from GDSC database.
    
    Contrary to the parent class Drug, this class also incorporates data related to targets
    derived from DrugBank, not only those from GDSC. Main function of the class is to create and store input data 
    corresponding to a given drug. Four types of data are considered: gene expression, copy number variants, 
    coding variants and tumor tissue type. Class instances are initialized with four basic drug properties: 
    ID, name, gene targets and target pathway. Data attributes are stored as pandas DataFrames and are filled 
    using data files from GDSC via corresponding methods.
    
    In general, all utilities are the same as in parent Drug class, with an exception of "create_drugs"
    method, which is overloaded in order to account for target genes data coming from DrugBank.
    
    Attributes:
        gdsc_id (int): ID from GDSC website.
        name (string): Drug name.
        targets (list of strings): Drug's target gene names (HGNC).
        target_pathway (string): Drug's target pathway as provided in GDSC annotations.
        ensembl targets (list of strings): Drug's target genes ensembl IDs. Can have different length 
                    than "targets" because some gene names may not be matched during mapping. Ensembl IDs are
                    needed for gene expression data.
        map_from_hgnc_to_ensembl (dictionary): Dictionary mapping from gene names to ensembl IDs. Created 
                    after calling the "load_mappings" method.
        map_from_ensembl_to_hgnc (dictionary): Dictionary mapping from ensembl IDs to gene names. Created 
                    after calling the "load_mappings" method.
        total_no_samples_screened (int): Number of cell lines screened for that drug. Created after 
                    calling the "extract_drug_response_data" method.
        response_data (DataFrame): DataFrame with screened cell lines for that drug and corresponding AUC or 
                    IC50 values. Created after calling the "extract_drug_response_data" method.
        screened_cell_lines (list of ints): list containing COSMIC IDs representing cell lines screened for 
                    that drug. Created after calling the "extract_screened_cell_lines" method.
        gene_expression_data (DataFrame): DataFrame with gene expression data, considering only
                    target genes. Created after calling the "extract_gene_expression" method
        mutation_data (DataFrame): DataFrame with binary calls for coding variants, considering only
                    target genes. Created after calling the "extract_mutation_data" method.
        cnv_data (DataFrame): DataFrame with binary calls for copu number variants, considering only
                    target genes. Created after calling the "extract_cnv_data" method.
        tissue_data (DataFrame): DataFrame with dummy encoded tumor tissue types in screened cell lines.
                    Dummy encoding results in 13 binary features. Created after calling the 
                    "extract_tissue_data" method.
        full_data (DataFrame): DataFrame with combined data coming from given set of genetic data
                    classes.
    
    Methods:
        Instance methods:
        __init__: Initialize a Drug instance.
        __repr__: Return string representation of an instance, as a command which can be used to create
                    this instance.
        __str__: Return string representation of an instance.
        extract_drug_response_data: Generate a DataFrame with drug-response data.
        extract_screened_cell_lines: Generate a list of COSMIC IDs representing cell lines screened for that
                    drug.
        extract_gene_expression: Generate a DataFrame with gene expression data for drug's screened cell lines
        extract_mutation_data: Generate a DataFrame with binary calls for coding variants.
        extract_cnv_data: Generate a DataFrame with binary calls for copy number variants.
        extract_tissue_data: Generate a DataFrame with dummy encoded tissue types.
        concatenate_data: Generate a DataFrame containing all desired genetic data classes. Available data
                    classes are: gene expression, coding variants, cnv variants and tissue type.
        create_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data.
        
        Class methods:
        load_mappings: Load appropriate dictionaries mapping between ensembl and HGNC.
        
        Static methods:
        create_drugs: Create a dictionary of DrugWithDrugBank class objects, each referenced by it's ID 
                    (keys are drug GDSC ID's). Includes also target data coming from DrugBank.
        load_data: Load all needed data files as DataFrames with one function call.
    
    """
    
    def create_drugs(drug_annotations_df, drugbank_targets_mapping):
        """Create a dictionary of DrugWithDrugBank class objects, each referenced by it's ID. Add
        also target data coming from DrugBank.

        Arguments:
            drug_annotations_df (DataFrame): DataFrame of drug annotations from GDSC website.
            drugbank_targets_mapping (dictionary): Dictionary with mapping from drug ID to it's
                    targets from drugbank database.

        Return:
            Dictionary of DrugWithDrugBank objects as values and their ID's as keys.
        """
        drugs = {}
        for row in drug_annotations_df.itertuples(index=True, name="Pandas"):
            name = getattr(row, "DRUG_NAME")
            gdsc_id = getattr(row, "DRUG_ID")
            targets = getattr(row, "TARGET").split(", ")
            # Add targets from DrugBank (if drug is matched) and take a sum
            if gdsc_id in drugbank_targets_mapping:
                targets = list(set(targets + drugbank_targets_mapping[gdsc_id]))
            target_pathway = getattr(row, "TARGET_PATHWAY")
            
            # Create DrugWithDrugBank instance and put it into output dictionary
            drugs[gdsc_id] = DrugWithDrugBank(gdsc_id, name, targets, target_pathway)
        return drugs
    
    def load_data(drug_annotations, cell_line_list, gene_expr, cnv1, cnv2, 
              coding_variants, drug_response, drugbank_targets):
        """Load all needed files by calling one function. All argumenst are filepaths to corrresponding files."""
        # Drug annotations
        drug_annotations_df = pd.read_excel(drug_annotations)
        # Cell line annotations
        col_names = ["Name", "COSMIC_ID", "TCGA classification", "Tissue", "Tissue_subtype", "Count"]
        cell_lines_list_df = pd.read_csv(cell_line_list, usecols=[1, 2, 3, 4, 5, 6], header=0, names=col_names)
        # Gene expression
        gene_expression_df = pd.read_table(gene_expr)
        # CNV
        d1 = pd.read_csv(cnv1)
        d2 = pd.read_table(cnv2)
        d2.columns = ["genes_in_segment"]
        def f(s):
            return s.strip(",")
        cnv_binary_df = d1.copy()
        cnv_binary_df["genes_in_segment"] = d2["genes_in_segment"].apply(f)
        # Coding variants
        coding_variants_df = pd.read_csv(coding_variants)
        # Drug-response
        drug_response_df = pd.read_excel(drug_response)
        # DrugBank targets
        map_drugs_to_drugbank_targets = pickle.load(open(drugbank_targets, "rb"))
        return (drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, coding_variants_df,
               drug_response_df, map_drugs_to_drugbank_targets)
    
#################################################################################################################
# DrugGenomeWide class
#################################################################################################################

class DrugGenomeWide(Drug):
    """Class designed to represent a drug with genome-wide input data. 
    
    Main function of the class is to create and store input data corresponding to a given
    drug. Four types of data are considered: gene expression, copy number variants, coding variants and tumor 
    tissue type. Class instances are initialized with four basic drug properties: ID, name, gene targets and
    target pathway. Data attributes are stored as pandas DataFrames and are filled using data files 
    from GDSC via corresponding methods.
    
    In general, all the utilities are the same as in the parent Drug class, but with different input data.
    When using this setting, we only use gene expression data as input, since it is recognized
    as representative of the genome-wide cell line characterization. Therefore, other data extraction methods,
    though available, should not be used when utilizing this class, for clarity. Two parent class 
    methods are overloaded: "extract_gene_expression" and "create_drugs".
    
    Important note: in here, "create_full_data" method is not overloaded, but is supposed to be called
    only parsing drug_response_df and gene_expression_df DataFrames and data_combination argument 
    set to "["expression"]".
    --Example:
        df = test_drug.create_full_data(drug_response_df, gene_expression_df, data_combination=["expression"])
        
    Attributes:
        gdsc_id (int): ID from GDSC website.
        name (string): Drug name.
        targets (list of strings): Drug's target gene names (HGNC).
        target_pathway (string): Drug's target pathway as provided in GDSC annotations.
        ensembl targets (list of strings): Drug's target genes ensembl IDs. Can have different length 
                    than "targets" because some gene names may not be matched during mapping. Ensembl IDs are
                    needed for gene expression data.
        map_from_hgnc_to_ensembl (dictionary): Dictionary mapping from gene names to ensembl IDs. Created 
                    after calling the "load_mappings" method.
        map_from_ensembl_to_hgnc (dictionary): Dictionary mapping from ensembl IDs to gene names. Created 
                    after calling the "load_mappings" method.
        total_no_samples_screened (int): Number of cell lines screened for that drug. Created after 
                    calling the "extract_drug_response_data" method.
        response_data (DataFrame): DataFrame with screened cell lines for that drug and corresponding AUC or 
                    IC50 values. Created after calling the "extract_drug_response_data" method.
        screened_cell_lines (list of ints): list containing COSMIC IDs representing cell lines screened for 
                    that drug. Created after calling the "extract_screened_cell_lines" method.
        gene_expression_data (DataFrame): DataFrame with gene expression data, considering all 
                    available (genome-wide) genes. Created after calling the "extract_gene_expression" 
                    method.
        mutation_data (DataFrame): DataFrame with binary calls for coding variants, considering only
                    target genes. Created after calling the "extract_mutation_data" method.
        cnv_data (DataFrame): DataFrame with binary calls for copu number variants, considering only
                    target genes. Created after calling the "extract_cnv_data" method.
        tissue_data (DataFrame): DataFrame with dummy encoded tumor tissue types in screened cell lines.
                    Dummy encoding results in 13 binary features. Created after calling the 
                    "extract_tissue_data" method.
        full_data (DataFrame): DataFrame with combined data coming from given set of genetic data
                    classes.
    
    Methods:
        Instance methods:
        __init__: Initialize a Drug instance.
        __repr__: Return string representation of an instance, as a command which can be used to create
                    this instance.
        __str__: Return string representation of an instance.
        extract_drug_response_data: Generate a DataFrame with drug-response data.
        extract_screened_cell_lines: Generate a list of COSMIC IDs representing cell lines screened for that
                    drug.
        extract_gene_expression: Generate a DataFrame with gene expression data for drug's screened cell lines
        extract_mutation_data: Generate a DataFrame with binary calls for coding variants.
        extract_cnv_data: Generate a DataFrame with binary calls for copy number variants.
        extract_tissue_data: Generate a DataFrame with dummy encoded tissue types.
        concatenate_data: Generate a DataFrame containing all desired genetic data classes. Available data
                    classes are: gene expression, coding variants, cnv variants and tissue type.
        create_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data. See the note above for correct
                    usage with DrugGenomeWide class.
        
        Class methods:
        load_mappings: Load appropriate dictionaries mapping between ensembl and HGNC.
        
        Static methods:
        create_drugs: Create a dictionary of Drug class objects, each referenced by it's ID 
                    (keys are drug GDSC ID's)
        load_data: Load all needed data files as DataFrames with one function call.
        
    """
    
    def extract_gene_expression(self, gene_expression_df):
        """Generate DataFrame of gene expression data for cell lines screened for this drug, 
        genome-wide (all available genes).
        
        Arguments:
            gene_expression_df (DataFrame): original GDSC gene expression DataFrame
            sensitivity_profiles_df (DataFrame): DataFrame of drug response data from GDSC
        
        Return:
            None
        """
        cell_lines_str = []   # Gene expression DF column names are strings
        for x in self.screened_cell_lines:
            cell_lines_str.append(str(x))
        cl_to_extract = []
        for x in cell_lines_str:
            if x in list(gene_expression_df.columns):   
                cl_to_extract.append(x)   # Extract only cell lines contained in gene expression data
        gene_expr = gene_expression_df[["ensembl_gene"] + cl_to_extract]
        gene_expr_t = gene_expr.transpose()
        columns = list(gene_expr_t.loc["ensembl_gene"])
        gene_expr_t.columns = columns
        gene_expr_t = gene_expr_t.drop(["ensembl_gene"])
        rows = list(gene_expr_t.index)
        gene_expr_t.insert(0, "cell_line_id", rows)   # Insert columns with cell line IDs
        gene_expr_t.reset_index(drop=True, inplace=True)
        gene_expr_t["cell_line_id"] = pd.to_numeric(gene_expr_t["cell_line_id"])
        # DataFrame should have same number of columns for each drug
        assert gene_expr_t.shape[1] == 17738
        self.gene_expression_data = gene_expr_t
        
    def extract_mutation_data(self, mutation_df):
        """Generate a DataFrame with binary mutation calls for screened cell lines and target genes.
        
        Arguments:
            mutation_df: DataFrame with original mutation calls from GDSC.
        
        Returns:
            None
        """
        targets = [x + "_mut" for x in self.targets]
        df = mutation_df.copy()[
                mutation_df.cosmic_sample_id.isin(self.screened_cell_lines)]
        df = df[["cosmic_sample_id", "genetic_feature", "is_mutated"]]
        cosmic_ids = []
        genetic_features = {}
        for feature in df.genetic_feature.unique():
            genetic_features[feature] = []
        for cl_id in df.cosmic_sample_id.unique():
            cosmic_ids.append(cl_id)
            df_cl = df[df.cosmic_sample_id == cl_id]
            for feature in genetic_features:
                mutation_status = df_cl[
                    df_cl.genetic_feature == feature]["is_mutated"].iloc[0]
                genetic_features[feature].append(mutation_status)
        df1 = pd.DataFrame()
        df1.insert(0, "cell_line_id", cosmic_ids)    # Insert column with samples IDs
        for feature in genetic_features:
            df1[feature] = genetic_features[feature]
        self.mutation_data = df1   # Put DataFrame into corresponding field
        
    def extract_cnv_data_faster(self, cnv_binary_df, map_cl_id_and_feature_to_status):
        """Generate data containing binary CNV calls for cell lines screened for the drug.
        
        Faster implementation than original "extract_cnv_data" by using mapping between genes and 
        genomic segments.
        
        Arguments:
            cnv_binary_df: DataFrame from GDSC download tool with CNV data.
        
        Returns:
            None
        """
        df = cnv_binary_df[cnv_binary_df.cosmic_sample_id.isin(self.screened_cell_lines)]
        features_to_extract = []   # Map drug's targets to CNV features (segments)
        
        cosmic_ids = []
        feature_dict = {}   # Separate lists for every column in final DataFrame
        for feature in df.genetic_feature.unique():
            feature_dict[feature] = []
        for cl_id in df.cosmic_sample_id.unique():
            cosmic_ids.append(cl_id)
            for feature in feature_dict:
                status = map_cl_id_and_feature_to_status[(cl_id, feature)]
                feature_dict[feature].append(status)
        new_df = pd.DataFrame()
        for feature in feature_dict:
            new_df[feature] = feature_dict[feature]
        new_df.insert(0, "cell_line_id", cosmic_ids)
        self.cnv_data = new_df
        
    def create_drugs(drug_annotations_df):
        """Create a dictionary of DrugGenomeWide class objects, each referenced by it's ID.

        Arguments:
        drug_annotations_df (DataFrame): DataFrame of drug annotations from GDSC website

        Returns:
        Dictionary of DrugGenomeWide objects as values and their ID's as keys
        """
        drugs = {}
        for row in drug_annotations_df.itertuples(index=True, name="Pandas"):
            gdsc_id = getattr(row, "DRUG_ID")
            name = getattr(row, "DRUG_NAME")
            targets = getattr(row, "TARGET").split(", ")
            target_pathway = getattr(row, "TARGET_PATHWAY")

            drugs[gdsc_id] = DrugGenomeWide(gdsc_id, name, targets, target_pathway)
        return drugs
    
    def load_data(drug_annotations, cell_line_list, gene_expr, cnv1, cnv2, 
              coding_variants, drug_response):
        """Load all needed files by calling one function. All argumenst are filepaths to corrresponding files."""
        # Drug annotations
        drug_annotations_df = pd.read_excel(drug_annotations)
        # Cell line annotations
        col_names = ["Name", "COSMIC_ID", "TCGA classification", "Tissue", "Tissue_subtype", "Count"]
        cell_lines_list_df = pd.read_csv(cell_line_list, usecols=[1, 2, 3, 4, 5, 6], header=0, names=col_names)
        # Gene expression
        gene_expression_df = pd.read_table(gene_expr)
        # CNV
        d1 = pd.read_csv(cnv1)
        d2 = pd.read_table(cnv2)
        d2.columns = ["genes_in_segment"]
        def f(s):
            return s.strip(",")
        cnv_binary_df = d1.copy()
        cnv_binary_df["genes_in_segment"] = d2["genes_in_segment"].apply(f)
        # Coding variants
        coding_variants_df = pd.read_csv(coding_variants)
        # Drug-response
        drug_response_df = pd.read_excel(drug_response)
        return (drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, coding_variants_df,
               drug_response_df)
    
#################################################################################################################
# DrugDirectReactome class
#################################################################################################################
    
class DrugDirectReactome(DrugWithDrugBank):
    """Class representing compound from GDSC database.
    
    Main function of the class is to create and store input data corresponding to a given
    drug. Four types of data are considered: gene expression, copy number variants, coding variants and tumor 
    tissue type. Class instances are initialized with four basic drug properties: ID, name, gene targets and
    target pathway. Data attributes are stored as pandas DataFrames and are filled using data files 
    from GDSC via corresponding methods.
    
    In this setting, drugs gene targets are derived not only from GDSC and DrugBank, but also using the direct
    compound-pathway mapping from Reactome database. All genes belonging to corresponding Reactome target pathway
    are considered when computing input data. The utilities are the same as in parent DrugWithDrugBank class with 
    an exception of "create_drugs" method which accounts for mappings coming from Reactome, and "load_data"
    method.
    
    Attributes:
        gdsc_id (int): ID from GDSC website.
        name (string): Drug name.
        targets (list of strings): Drug's target gene names (HGNC).
        target_pathway (string): Drug's target pathway as provided in GDSC annotations.
        ensembl targets (list of strings): Drug's target genes ensembl IDs. Can have different length 
                    than "targets" because some gene names may not be matched during mapping. Ensembl IDs are
                    needed for gene expression data.
        map_from_hgnc_to_ensembl (dictionary): Dictionary mapping from gene names to ensembl IDs. Created 
                    after calling the "load_mappings" method.
        map_from_ensembl_to_hgnc (dictionary): Dictionary mapping from ensembl IDs to gene names. Created 
                    after calling the "load_mappings" method.
        total_no_samples_screened (int): Number of cell lines screened for that drug. Created after 
                    calling the "extract_drug_response_data" method.
        response_data (DataFrame): DataFrame with screened cell lines for that drug and corresponding AUC or 
                    IC50 values. Created after calling the "extract_drug_response_data" method.
        screened_cell_lines (list of ints): list containing COSMIC IDs representing cell lines screened for 
                    that drug. Created after calling the "extract_screened_cell_lines" method.
        gene_expression_data (DataFrame): DataFrame with gene expression data, considering only
                    target genes. Created after calling the "extract_gene_expression" method
        mutation_data (DataFrame): DataFrame with binary calls for coding variants, considering only
                    target genes. Created after calling the "extract_mutation_data" method.
        cnv_data (DataFrame): DataFrame with binary calls for copu number variants, considering only
                    target genes. Created after calling the "extract_cnv_data" method.
        tissue_data (DataFrame): DataFrame with dummy encoded tumor tissue types in screened cell lines.
                    Dummy encoding results in 13 binary features. Created after calling the 
                    "extract_tissue_data" method.
        full_data (DataFrame): DataFrame with combined data coming from given set of genetic data
                    classes.
    
    Methods:
        Instance methods:
        __init__: Initialize a Drug instance.
        __repr__: Return string representation of an instance, as a command which can be used to create
                    this instance.
        __str__: Return string representation of an instance.
        extract_drug_response_data: Generate a DataFrame with drug-response data.
        extract_screened_cell_lines: Generate a list of COSMIC IDs representing cell lines screened for that
                    drug.
        extract_gene_expression: Generate a DataFrame with gene expression data for drug's screened cell lines
        extract_mutation_data: Generate a DataFrame with binary calls for coding variants.
        extract_cnv_data: Generate a DataFrame with binary calls for copy number variants.
        extract_tissue_data: Generate a DataFrame with dummy encoded tissue types.
        concatenate_data: Generate a DataFrame containing all desired genetic data classes. Available data
                    classes are: gene expression, coding variants, cnv variants and tissue type.
        create_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data.
        
        Class methods:
        load_mappings: Load appropriate dictionaries mapping between ensembl and HGNC.
        
        Static methods:
        create_drugs: Create a dictionary of DrugWithDrugBank class objects, each referenced by it's ID 
                    (keys are drug GDSC ID's). Includes also target data coming from DrugBank.
        load_data: Load all needed data files as DataFrames with one function call.
                    
        """
    
    def create_drugs(drug_annotations_df, drugbank_targets_mapping, reactome_direct_mapping):
        """Create a dictionary of DrugWithDrugBank class objects, each referenced by it's ID.

        Arguments:
        drug_annotations_df (DataFrame): DataFrame of drug annotations from GDSC website
        drugbank_targets_mapping (dictionary): dictionary with mapping from drug ID to it's
                targets from drugbank database
        reactome_direct_mapping: 

        Returns:
        Dictionary of Drug objects as values and their ID's as keys
        """
        drugs = {}
        for row in drug_annotations_df.itertuples(index=True, name="Pandas"):
            name = getattr(row, "DRUG_NAME")
            gdsc_id = getattr(row, "DRUG_ID")
            
            # Create an object only if it exists in Reactome mapping dictionary
            if gdsc_id in reactome_direct_mapping:
                targets = getattr(row, "TARGET").split(", ")
                # If this ID exists in DrugBank mapping, take the sum of all three sets
                if gdsc_id in drugbank_targets_mapping:
                    targets = list(set(targets + drugbank_targets_mapping[gdsc_id] + reactome_direct_mapping[gdsc_id]))
                # Otherwise add just the Reactome targets
                else:
                    targets = list(set(targets + reactome_direct_mapping[gdsc_id]))
                target_pathway = getattr(row, "TARGET_PATHWAY")
                drugs[gdsc_id] = DrugDirectReactome(gdsc_id, name, targets, target_pathway)
            else:
                continue
        return drugs
    
    def load_data(drug_annotations, cell_line_list, gene_expr, cnv1, cnv2, 
              coding_variants, drug_response, drugbank_targets, reactome_targets):
        """Load all needed files by calling one function. All argumenst are filepaths to 
        corrresponding files."""
        # Drug annotations
        drug_annotations_df = pd.read_excel(drug_annotations)
        # Cell line annotations
        col_names = ["Name", "COSMIC_ID", "TCGA classification", "Tissue", "Tissue_subtype", "Count"]
        cell_lines_list_df = pd.read_csv(cell_line_list, usecols=[1, 2, 3, 4, 5, 6], header=0, names=col_names)
        # Gene expression
        gene_expression_df = pd.read_table(gene_expr)
        # CNV
        d1 = pd.read_csv(cnv1)
        d2 = pd.read_table(cnv2)
        d2.columns = ["genes_in_segment"]
        def f(s):
            return s.strip(",")
        cnv_binary_df = d1.copy()
        cnv_binary_df["genes_in_segment"] = d2["genes_in_segment"].apply(f)
        # Coding variants
        coding_variants_df = pd.read_csv(coding_variants)
        # Drug-response
        drug_response_df = pd.read_excel(drug_response)
        # DrugBank targets
        map_drugs_to_drugbank_targets = pickle.load(open(drugbank_targets, "rb"))
        # Reactome targets
        map_drugs_to_reactome_targets = pickle.load(open(reactome_targets, "rb"))
        return (drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, coding_variants_df,
               drug_response_df, map_drugs_to_drugbank_targets, map_drugs_to_reactome_targets)
    
#################################################################################################################
# DrugWithGenesInSamePathways class
#################################################################################################################
    
class DrugWithGenesInSamePathways(DrugWithDrugBank):
    """Class representing drug from GDSC database.
    
    Main function of the class is to create and store input data corresponding to a given
    drug. Four types of data are considered: gene expression, copy number variants, coding variants and tumor 
    tissue type. Class instances are initialized with four basic drug properties: ID, name, gene targets and
    target pathway. Data attributes are stored as pandas DataFrames and are filled using data files 
    from GDSC via corresponding methods.
    
    In general, all the utilities are the same as in the basic Drug class, but with different input data.
    As in DrugWithDrugBank, we inorporate genetic features related to target genes coming both from GDSC and 
    DrugBank, but in addition to this, we also consider every gene that belongs to the same pathways as original
    target genes. In other words, we consider genes that belong to pathways which are picked based on a single 
    membership of any of the target genes.
    
    Three of original methods are overloaded: __init__, "create_drugs" and "load_data". Furthermore, three new 
    attributes are introduced: "original_targets" which stores actual targets coming from GDSC and DrugBank, and 
    "related_genes" which stores genes that occur in the same pathways along with original targets.
    
    Attributes:
        gdsc_id (int): ID from GDSC website.
        name (string): Drug name.
        targets (list of strings): Drug's target gene names (HGNC).
        target_pathway (string): Drug's target pathway as provided in GDSC annotations.
        ensembl targets (list of strings): Drug's target genes ensembl IDs. Can have different length 
                    than "targets" because some gene names may not be matched during mapping. Ensembl IDs are
                    needed for gene expression data.
        map_from_hgnc_to_ensembl (dictionary): Dictionary mapping from gene names to ensembl IDs. Created 
                    after calling the "load_mappings" method.
        map_from_ensembl_to_hgnc (dictionary): Dictionary mapping from ensembl IDs to gene names. Created 
                    after calling the "load_mappings" method.
        total_no_samples_screened (int): Number of cell lines screened for that drug. Created after 
                    calling the "extract_drug_response_data" method.
        response_data (DataFrame): DataFrame with screened cell lines for that drug and corresponding AUC or 
                    IC50 values. Created after calling the "extract_drug_response_data" method.
        screened_cell_lines (list of ints): list containing COSMIC IDs representing cell lines screened for 
                    that drug. Created after calling the "extract_screened_cell_lines" method.
        gene_expression_data (DataFrame): DataFrame with gene expression data, considering only
                    target genes. Created after calling the "extract_gene_expression" method
        mutation_data (DataFrame): DataFrame with binary calls for coding variants, considering only
                    target genes. Created after calling the "extract_mutation_data" method.
        cnv_data (DataFrame): DataFrame with binary calls for copu number variants, considering only
                    target genes. Created after calling the "extract_cnv_data" method.
        tissue_data (DataFrame): DataFrame with dummy encoded tumor tissue types in screened cell lines.
                    Dummy encoding results in 13 binary features. Created after calling the 
                    "extract_tissue_data" method.
        full_data (DataFrame): DataFrame with combined data coming from given set of genetic data
                    classes.
    
    Methods:
        Instance methods:
        __init__: Initialize a Drug instance.
        __repr__: Return string representation of an instance, as a command which can be used to create
                    this instance.
        __str__: Return string representation of an instance.
        extract_drug_response_data: Generate a DataFrame with drug-response data.
        extract_screened_cell_lines: Generate a list of COSMIC IDs representing cell lines screened for that
                    drug.
        extract_gene_expression: Generate a DataFrame with gene expression data for drug's screened cell lines
        extract_mutation_data: Generate a DataFrame with binary calls for coding variants.
        extract_cnv_data: Generate a DataFrame with binary calls for copy number variants.
        extract_tissue_data: Generate a DataFrame with dummy encoded tissue types.
        concatenate_data: Generate a DataFrame containing all desired genetic data classes. Available data
                    classes are: gene expression, coding variants, cnv variants and tissue type.
        create_full_data: Combines above data extraction methods in order to create desired input data
                    for the drug with one method call. Returns the full data.
        
        Class methods:
        load_mappings: Load appropriate dictionaries mapping between ensembl and HGNC.
        
        Static methods:
        create_drugs: Create a dictionary of DrugWithDrugBank class objects, each referenced by it's ID 
                    (keys are drug GDSC ID's). Includes also target data coming from DrugBank.
        load_data: Load all needed data files as DataFrames with one function call.
        
        """
    
    def __init__(self, gdsc_id, name, original_targets, related_genes, targets, target_pathway):
        
        self.gdsc_id = gdsc_id
        self.name = name
        self.original_targets = original_targets
        self.related_genes = related_genes
        self.targets = targets
        self.target_pathway = target_pathway
        self.ensembl_targets = []
        for x in self.targets:
            try:
                self.ensembl_targets.append(self.map_from_hgnc_to_ensembl[x])
            except KeyError:
                pass
    
    def create_drugs(drug_annotations_df, drugbank_targets_mapping, 
                     map_target_genes_to_same_pathways_genes):
        """Create a dictionary of DrugWithGenesInSamePathways class objects, each referenced by it's ID. Add
        target data coming from DrugBank, as well as genes that occur in the same pathways as original target
        genes.

        Arguments:
            drug_annotations_df (DataFrame): DataFrame of drug annotations from GDSC website.
            drugbank_targets_mapping (dictionary): Dictionary with mapping from drug ID to it's
                    targets from drugbank database.
            map_target_genes_to_same_pathways_genes (dictionary): Dictionary with mapping from target
                    genes names to gene names that occur in same pathways.

        Returns:
            Dictionary of DrugWithGenesInSamePathways objects as values and their ID's as keys.
        """
        drugs = {}
        for row in drug_annotations_df.itertuples(index=True, name="Pandas"):
            # Extract name, ID and GDSC target pathway
            name = getattr(row, "DRUG_NAME")
            gdsc_id = getattr(row, "DRUG_ID")
            target_pathway = getattr(row, "TARGET_PATHWAY")
            
            # Extract original targets (for now just from GDSC)
            original_targets = getattr(row, "TARGET").split(", ")
            # Add genes from DrugBank (if drug is matched) to original targets list
            if gdsc_id in drugbank_targets_mapping:
                original_targets = list(set(original_targets + drugbank_targets_mapping[gdsc_id]))
                
            # Extract list of related genes (that occur in same pathways but are not direct targets)
            # from Reactome dictionary
            related_genes = []
            # Iterate over direct target genes
            for target_gene in original_targets:
                if target_gene in map_target_genes_to_same_pathways_genes:
                    # Genes related to one specific target
                    coocurring_genes = map_target_genes_to_same_pathways_genes[target_gene]
                    # Update the overall list of related genes for given drug
                    related_genes = related_genes + coocurring_genes
            # Exclude genes that are original targets fro related genes
            related_genes_final = []
            for gene in related_genes:
                if gene not in original_targets:
                    related_genes_final.append(gene)
            # Remove duplicates from related_genes_final
            related_genes_final = list(set(related_genes_final))
            
            # Setup "targets" field as a sum of original targets and related genes
            targets = original_targets + related_genes_final
            assert len(targets) == len(original_targets) + len(related_genes_final)
            
            # Finally, create actual DrugWithGenesInSamePathways instance
            drugs[gdsc_id] = DrugWithGenesInSamePathways(gdsc_id, name, original_targets,
                                            related_genes_final, targets, target_pathway)
        return drugs
            
            
    def load_data(drug_annotations, cell_line_list, gene_expr, cnv1, cnv2, 
              coding_variants, drug_response, drugbank_targets, pathway_occurence_genes):
        """Load all needed files by calling one function. All argumenst are filepaths to corrresponding files."""
        # Drug annotations
        drug_annotations_df = pd.read_excel(drug_annotations)
        # Cell line annotations
        col_names = ["Name", "COSMIC_ID", "TCGA classification", "Tissue", "Tissue_subtype", "Count"]
        cell_lines_list_df = pd.read_csv(cell_line_list, usecols=[1, 2, 3, 4, 5, 6], header=0, names=col_names)
        # Gene expression
        gene_expression_df = pd.read_table(gene_expr)
        # CNV
        d1 = pd.read_csv(cnv1)
        d2 = pd.read_table(cnv2)
        d2.columns = ["genes_in_segment"]
        def f(s):
            return s.strip(",")
        cnv_binary_df = d1.copy()
        cnv_binary_df["genes_in_segment"] = d2["genes_in_segment"].apply(f)
        # Coding variants
        coding_variants_df = pd.read_csv(coding_variants)
        # Drug-response
        drug_response_df = pd.read_excel(drug_response)
        # DrugBank targets
        map_drugs_to_drugbank_targets = pickle.load(open(drugbank_targets, "rb"))
        # Dictionary mapping from target genes to genes that occur in same pathways
        map_target_genes_to_same_pathways_genes = pickle.load(
        open(pathway_occurence_genes, "rb"))
        return (drug_annotations_df, cell_lines_list_df, gene_expression_df, cnv_binary_df, coding_variants_df,
               drug_response_df, map_drugs_to_drugbank_targets, map_target_genes_to_same_pathways_genes)
    
#################################################################################################################
# Experiment class
#################################################################################################################

class Experiment(object):
    """Class representing single machine learning experiment with GDSC data.
    
    The class is mainly useful for storing experiment's results. Actual machine learning 
    is performed outside of the class. Data stored in dictonaries is filled during machine learning.
    Other methods describing the results use those primal data dictonaries.
    
    Attributes:
        name (string): Name somehow summarizing the experiment.
        algorithm (string): Name of the predictive algorithm used.
        param_search_type (string): What kind of parameter search was used: exhaustive grid search
                    or randomized search.
        data_normalization_type (string): What kind of data normalization was applied, on which features.
        kfolds (int): How many folds were used during cross-validation hyperparameter tuning
        split_seed (int): Seed (random_state) used when randomly splitting the data (using sklearn's 
                    train_test_split). If None, no specific seed was set.
        tuning_seed (int): Seed (random_state) used when doing hyperparameter tuning with sklearn's
                    RandomizedSearchCV.
        input_data (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and DataFrames
                    containing corrresponding input data as values
        best_scores (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and tuples of 
                    achieved best prediction scores as values.
        cv_results: (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and full cross-validation
                    results as values
        best_parameters (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and best found 
                    hyperparameters as values
        dummy_scores (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and corresponding
                    dummy scores as values
        data_stds (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and standard deviations
                    of response variable as values
        coefficients (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and weights of the best
                    found model as values.
        training_scores (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and scores 
                    obtained on the training set as values.
        data_shapes (dictionary): Dictionary with (drug.name, drug.gdsc_id) pairs as keys and tuples of input
                    data shapes as values.
        results_df (DataFrame): DataFrame with results of the Experiment for each considered drug.
        
    
    Methods:
        Instance methods:
        __init__: Initializer.
        __repr__: Return short string representation of an instance.
        __str__: Return more detailed description of an experiment.
        create_results_df: Wrap results from dictonaries into single pandas DataFrame
        results_summary_single: Display numeric summary of Experiment's results.
        boxplot_of_performance_single: Create boxplot of a given perfromance metric across all drugs.
        barplot_of_rmses_single: Generate barplots of model test RMSEs along with dummy RMSE for each drug, 
                    for single Experiment.
        boxplots_rel_rmse_wrt_pathways: Generate boxplots of relative RMSE wrt. target pathways.
        plot_one_vs_another: Create regular plot of first variable vs. another.
        plot_pathway_distribution: Plot overall pathway counts in this Experiment.
        list_of_better_drugs: Compute list of drugs which performed better in the "other" Experiment.
        plot_pathway_distribution_in_better_drugs: Plot target pathway counts in better performing drugs
                    in "other" Experiment.
        merge_two_exp_results: Take other Eperiment object and horizontally merge two corresponding result
                    DataFrames.
        results_for_better_drugs: Compute DataFrame containing comparisons between two Experiments along 
                    drugs that performed better in the other one.
        barplots_of_rmse_for_two_experiments: Plot barplots of test RMSE or relative test RMSE for each drug. 
        boxplots_of_performance_for_two_experiments: Create boxplots of a given performance metric for two 
                    Experiments.
        
        Static methods:
        create_input_for_each_drug: Take in the dictionary with drug_id: Drug object pairs and create 
        input data for each Drug.
        """
    
    # Instance methods
    def __init__(self, name, algorithm, parameter_search_type, data_normalization_type,
                kfolds, split_seed=None, tuning_seed=None):
        """Experiment class initializer.
        
        Attributes other than name will be filled during actual learning.
        
        Arguments:
            name (string): Name somehow summarizing the experiment.
            algorithm (string): Name of the predictive algorithm used.
            param_search_type (string): What kind of parameter search was used: exhaustive grid search
                        or randomized search.
            data_normalization_type (string): What kind of data normalization was applied, on which features.
            kfolds (int): How many folds were used during cross-validation hyperparameter tuning
            split_seed (int): Seed (random_state) used when randomly splitting the data (using sklearn's 
                        train_test_split). If None, no specific seed was set.
            tuning_seed (int): Seed (random_state) used when doing hyperparameter tuning with sklearn's
                        RandomizedSearchCV.
        
        Returns:
            None
        """
        # General characteristics
        self.name = name
        self.algorithm = algorithm
        self.param_search_type = parameter_search_type
        self.data_normalization_type = data_normalization_type
        self.kfolds = kfolds
        self.split_seed = split_seed
        self.tuning_seed = tuning_seed
        
        # Results storage dictonaries
        self.input_data = {}
        self.best_scores = {}
        self.cv_results = {}
        self.best_parameters = {} 
        self.dummy_scores = {}
        self.data_stds = {}
        self.coefficients = {}
        self.training_scores = {}
        self.data_shapes = {}
        
    def __repr__(self):
        """Return short string representation of an object."""
        return 'Experiment("{}")'.format(self.name)
    
    def __str__(self):
        """Return more detailed description of an experiment."""
        return self.name
        
    def create_results_df(self, drug_annotations_df):
        """Wrap results in storage dictionaries into single DataFrame.
        
        Assign resulting DataFrame into instance attribute (results_df).
        
        Arguments:
        drug_annotations_df (DataFrame): list of drugs in GDSC
        
        Returns:
        None
        """
        # Initialize DataFrame to fill
        df = pd.DataFrame()
        # Initialize lists (columns) we want to have
        drug_names = []
        drug_ids = []
        model_test_RMSES = []
        model_CV_RMSES = []
        model_test_correlations = []
        model_test_corr_pvals = []
        dummy_test_RMSES = []
        dummy_CV_RMSES = []
        y_stds = []
        model_train_RMSES = []
        numbers_of_samples = []
        numbers_of_features = []
        target_pathways = []

        # Fill the columns lists
        for name, ide in self.best_scores:
            drug_names.append(name)
            drug_ids.append(ide)
            model_test_RMSES.append(self.best_scores[(name, ide)].test_RMSE)
            model_CV_RMSES.append((-self.best_scores[(name, ide)].cv_best_score) ** 0.5)
            model_test_correlations.append(self.best_scores[(name, ide)].test_correlation[0])
            model_test_corr_pvals.append(self.best_scores[(name, ide)].test_correlation[1])
            dummy_test_RMSES.append(self.dummy_scores[(name, ide)].test_RMSE)
            dummy_CV_RMSES.append(self.dummy_scores[(name, ide)].cv_RMSE)
            y_stds.append(self.data_stds[(name, ide)].overall)
            model_train_RMSES.append(self.training_scores[(name, ide)].training_RMSE)
            numbers_of_samples.append(self.data_shapes[(name, ide)][0])
            numbers_of_features.append(self.data_shapes[(name, ide)][1] - 14)
            target_pathways.append(
                drug_annotations_df[drug_annotations_df["DRUG_ID"] == ide]["TARGET_PATHWAY"].iloc[0])

        # Insert lists as a DataFrame columns
        df.insert(0, "Drug Name", drug_names)
        df.insert(1, "Drug ID", drug_ids)
        df.insert(2, "Target Pathway", target_pathways)
        df.insert(3, "Model test RMSE", model_test_RMSES)
        df.insert(4, "Model CV RMSE", model_CV_RMSES)
        df.insert(5, "Model test correlation", model_test_correlations)
        df.insert(6, "Model test pval", model_test_corr_pvals)
        df.insert(7, "Dummy test RMSE", dummy_test_RMSES)
        df.insert(8, "Dummy CV RMSE", dummy_CV_RMSES)
        df.insert(9, "STD in Y", y_stds)
        df.insert(10, "Model train RMSE", model_train_RMSES)
        df.insert(11, "Number of samples", numbers_of_samples)
        df.insert(12, "Number of features", numbers_of_features)
        df.insert(13, "Relative test RMSE", df["Dummy test RMSE"] / df["Model test RMSE"])

        self.results_df = df
        
    def results_summary_single(self):
        """Display numeric summary of Experiment's results.
        
        Arguments:
        None
        
        Returns:
        None
        """
        print(self.name)
        print("Mean and median test RMSE:", round(self.results_df["Model test RMSE"].mean(), 4),
             round(self.results_df["Model test RMSE"].median(), 4))
        print('Mean and median test correlation:', round(self.results_df["Model test correlation"].mean(), 4),
             round(self.results_df["Model test correlation"].median(), 4))
        
        
    # Plots of result from single Experiment
    def boxplot_of_performance_single(self, metric, title="Predictive performance", figsize = (6, 6), 
                                      title_size=25, label_size=20, 
                                      tick_size=20, save_directory=None):
        """Creates boxplot of a given perfromance metric across all drugs.
        
        Arguments:
        metric (string): which variable to plot, must be one of the column names in self.results_df
        title (string): title of the plot
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
        
        Returns:
        None
        """
        fig = plt.figure(figsize=figsize)
        plt.tick_params(labelsize=tick_size)
        plt.title(title, fontsize=title_size)
        plt.grid()
        
        sns.boxplot(x=self.results_df[metric], orient="v")
        plt.xlabel("", fontsize=label_size)
        plt.ylabel(metric, fontsize=label_size)
        
        if save_directory:
            plt.savefig(save_directory, bbox_inches='tight')
        plt.show()
        
        
    def barplot_of_rmses_single(self, figsize=(35, 12), title_size=45, label_size=30, tick_size=20,
                        half=1, save_directory=None):
        """Generate barplots of model test RMSEs along with dummy RMSE for each drug, for single
        Experiment.
        
        Arguments:
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        half (int): which half of the drugs to plot. If 0, all available drugs are plotted
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
        
        Returns:
        None
        """
        # Set up legend parameters
        params = {"legend.fontsize": 25,
                 "legend.handlelength": 2}
        plt.rcParams.update(params)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Cross-validation RMSE for each drug", fontsize=title_size)
        ax.set_xlabel("Drug", fontsize=label_size)
        ax.set_ylabel("RMSE", fontsize=label_size)
    
        # Set up DataFrame slicing
        if half == 1:
            start_idx = 0
            end_idx = self.results_df.shape[0] // 2
        elif half == 2:
            start_idx = self.results_df.shape[0] // 2
            end_idx = self.results_df.shape[0]
        else:
            start_idx = 0
            end_idx = self.results_df.shape[0]

        self.results_df.sort_values("Model test RMSE", ascending = False).iloc[start_idx:end_idx].plot(
            x = "Drug Name", y = ["Model test RMSE", "Dummy test RMSE"] ,kind="bar", ax = ax, width = 0.7,
            figsize=figsize, fontsize=tick_size, legend = True, grid = True)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    def boxplots_rel_rmse_wrt_pathways(self, figsize=(30, 12), title_size=35, label_size=30,
                                      tick_size=25, save_directory=None):
        """Generate boxplots of relative RMSE wrt. target pathways.
        
        Arguments:
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
            
        Returns:
        None
        """
        # Set up order list for plotting
        order_list = list(self.results_df.groupby("Target Pathway").agg({"Relative test RMSE": np.median}).sort_values(
                    "Relative test RMSE").sort_values( "Relative test RMSE", ascending = False).index)
        fig = plt.figure(figsize=figsize)
        plt.tick_params(labelsize=tick_size)
        plt.xticks(rotation="vertical")
        plt.title("Relative test RMSE with respect to drug's target pathway", fontsize=title_size)
        plt.grid()
        plt.gcf().subplots_adjust(bottom=0.53)

        sns.boxplot(x = "Target Pathway", y = "Relative test RMSE", data = self.results_df, order=order_list)

        plt.ylabel("Relative test RMSE", fontsize=label_size)
        plt.xlabel("Target pathway", fontsize=label_size)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
    
    def plot_one_vs_another(self, first, second, title, hline=False, figsize=(15, 10), title_size=30, label_size=25,
                                      tick_size=15, save_directory=None):
        """Create regular plot of first variable vs. another.
        
        Arguments:
        first (string): first variable to plot, must be one of the column names in self.results_df
        second(string): second variable to plot, must be one of the column names in self.results_df
        hline (bool): do we want a horizontal line at one
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
            
        Returns:
        None
        """
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=title_size)
        plt.xlabel(second, fontsize=label_size)
        plt.ylabel(first, fontsize=label_size)
        plt.tick_params(labelsize = tick_size)
        
        if hline:
            plt.axhline(y=1.0, xmin=0, xmax=1.2, color="black", linewidth=2.5)
        sns.regplot(x=second, y=first, data=self.results_df)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    def plot_pathway_distribution(self, figsize=(15, 8), title_size=30, label_size=25,
                                      tick_size=15, save_directory=None):
        """Plot overall pathway counts in this Experiment.
        
        Arguments:
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
            
        Returns:
        None
        """
        plt.figure(figsize=figsize)
        plt.title("Target Pathway counts", fontsize=title_size)
        plt.xlabel("Target Pathway", fontsize=label_size)
        
        sns.countplot(x="Target Pathway", data=self.results_df)
        
        plt.ylabel("Count", fontsize=label_size)
        plt.xticks(rotation='vertical')
        plt.tick_params("both", labelsize=tick_size)
        plt.gcf().subplots_adjust(bottom=0.45)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
        
    
    # Comparisons between two Experiments
    def list_of_better_drugs(self, other, rmse_type="relative", rmse_fraction=0.8, 
                             correlation_fraction=None, correlation_threshold=0.25):
        """Compute list of drugs which performed better in the "other" experiment.
        
        Arguments:
        other (Experiment): other considered Experiment object
        rmse_type (string): "relative" or "absolute" - whether to use relative or absolute RMSE in comparison
        rmse_fraction (float): fraction by which other RMSE should be better for drug to
                be considered as "better". If rmse_type is "relative", fraction should be greater than 1.,
                less than 1. otherwise.
        correlation_fraction (float, default None): fraction by which other correlation should be better for 
                drug to be considered as "better". Should be greater thatn 1. If None, this condidion is not applied.
        correlation_threshold (float): correlation that needs to be achieved in order
                for drug to be "better"
                
        Returns:
        list if better drugs IDs
        """
        if rmse_type == "relative":
            column = "Relative test RMSE"
        if rmse_type == "absolute":
            column = "Model test RMSE"
            
        better_drugs = []
        for drug_id in self.results_df["Drug ID"].unique():
            self_df = self.results_df[self.results_df["Drug ID"] == drug_id]
            other_df = other.results_df[other.results_df["Drug ID"] == drug_id]
            if other_df.shape[0] < 1:   # Make sure of other Experiment contains results for this drug
                continue
            # Extract data of interest
            self_rmse = self_df[column].iloc[0]
            self_corr = self_df["Model test correlation"].iloc[0]
            other_rmse = other_df[column].iloc[0]
            other_corr = other_df["Model test correlation"].iloc[0]
            other_relative = other_df["Relative test RMSE"].iloc[0]
            
            # Classify as good or bad
            if rmse_type == "relative":   # Need to distungish two cases because the higher relative RMSE the better
                if correlation_fraction:
                    if (other_rmse > rmse_fraction * self_rmse) and (other_relative > 1.) \
                    and (other_corr > correlation_threshold) and (other_corr > correlation_fraction * self_corr):
                        better_drugs.append(drug_id)
                else:
                    if (other_rmse > rmse_fraction * self_rmse) and (other_relative > 1.) \
                    and (other_corr > correlation_threshold):
                        better_drugs.append(drug_id)
            if rmse_type == "absolute":   # The lower absolute RMSE the better
                if correlation_fraction:
                    if (other_rmse < rmse_fraction * self_rmse) and (other_relative > 1.) \
                    and (other_corr > correlation_threshold) and (other_corr > correlation_fraction * self_corr):
                        better_drugs.append(drug_id)
                else:
                    if (other_rmse > rmse_fraction * self_rmse) and (other_relative > 1.) \
                    and (other_corr > correlation_threshold):
                        better_drugs.append(drug_id)
        return better_drugs
    
    def plot_pathway_distribution_in_drugs_intersection(self, other, figsize=(15, 8), title_size=30, label_size=25,
                                      tick_size=15, save_directory=None):
        """Plot pathway distribution in this (self) Experiment across drugs, but only consider drugs common
        with "other" Experiment.
        
        Arguments:
        other (Experiment): other considered Experiment object
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
            
        Returns:
        None
        """
        # Compute intersection in IDs
        intersection = list(self.merge_two_exp_results(
                            other, ["a", "b"], "flag")["Drug ID"].unique())
        # Actual plot
        plt.figure(figsize=figsize)
        plt.title("Target Pathway counts", fontsize=title_size)
        plt.xlabel("Target Pathway", fontsize=label_size)
        
        sns.countplot(x="Target Pathway", data=self.results_df[self.results_df["Drug ID"].isin(intersection)])
        
        plt.ylabel("Count", fontsize=label_size)
        plt.xticks(rotation='vertical')
        plt.tick_params("both", labelsize=tick_size)
        plt.gcf().subplots_adjust(bottom=0.45)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
            
            
    # Plots of results from two Experiments
    def plot_pathway_distribution_in_better_drugs(self, other, rmse_type="relative",
                                                  rmse_fraction=0.8, correlation_fraction=None,
                                                  correlation_threshold=0.25,
                                                  figsize=(15, 8), title_size=30, 
                                                  label_size=25, tick_size=15, save_directory=None):
        """Plot target pathway counts in better performing drugs.
        
        Arguments:
        other (Experiment): other considered Experiment object
        rmse_type (string): "relative" or "absolute" - whether to use relative or 
                absolute RMSE in comparison
        rmse_fraction (float): fraction by which other RMSE should be better for drug to
                be considered as "better". If rmse_type is "relative", fraction should be greater than 1.,
                less than 1. otherwise.
        correlation_fraction (float, default None): fraction by which other correlation should be better for 
                drug to be considered as "better". Should be greater thatn 1. If None, this condidion is not applied.
        correlation_threshold (float): correlation that needs to be achieved in order
                for drug to be "better"
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
            
        Returns:
        None
        """
        # Compute list of better drugs
        better_drugs = self.list_of_better_drugs(other, rmse_type, rmse_fraction, correlation_fraction,
                                                    correlation_threshold)
        plt.figure(figsize=figsize)
        plt.title("Target Pathway counts", fontsize=title_size)
        plt.xlabel("Target Pathway", fontsize=label_size)
        plt.ylabel("Count", fontsize=label_size)
        
        sns.countplot(x="Target Pathway", data=other.results_df[other.results_df["Drug ID"].isin(better_drugs)
        ])
        
        plt.xticks(rotation='vertical')
        plt.tick_params("both", labelsize=tick_size)
        plt.gcf().subplots_adjust(bottom=0.45)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    
    def merge_two_exp_results(self, other, flags, feature_name):
        """Take other Eperiment object and horizontally merge two corresponding result
        DataFrames.
        
        Arguments:
        other (Experiment): other considered Experiment object
        flags (list of strings): flags assigned to DataFrame entries from given object. First element 
                should be for this (self) instance
        feature_name (string): name of a column that differentiate between two results sets
        
        Returns:
        None
        """
        # Since the whole point is to compare results for the same drugs, first let's find
        # intersection between two DataFrames in terms of Drug ID
        ids_intersection = list(self.results_df[
            self.results_df["Drug ID"].isin(list(other.results_df["Drug ID"].unique()))]["Drug ID"].unique())
        # Create intersected DataFrames
        self_intersected_df = self.results_df[self.results_df["Drug ID"].isin(ids_intersection)]
        other_intersected_df = other.results_df[other.results_df["Drug ID"].isin(ids_intersection)]
        assert self_intersected_df.shape[0] == other_intersected_df.shape[0]
        
        # Create new DataFrame with merged results
        # First, assign flags to self DataFrame entries
        flag_list = [flags[0]] * self_intersected_df.shape[0]
        self_intersected_df[feature_name] = flag_list
        
        # Second DataFrame
        flag_list = [flags[1]] * other_intersected_df.shape[0]
        other_intersected_df[feature_name] = flag_list
        
        # Concatenate both DataFrames
        total_results_df = pd.concat([self_intersected_df, other_intersected_df], axis=0)
        return total_results_df
    
    
    def results_for_better_drugs(self, other, flags, feature_name, rmse_type="relative",
                                 rmse_fraction=0.8, correlation_fraction=None, correlation_threshold=0.2):
        """Compute DataFrame containing comparisons between two Experiments along drugs
        that performed better in the other one.
        
        Arguments:
        other (Experiment): other considered Experiment object
        flags (list of strings): flags assigned to DataFrame entries from given object. First element 
                should be for this (self) instance
        feature_name (string): name of a column that differentiate between two results sets
        rmse_type (string): "relative" or "absolute" - whether to use relative or absolute RMSE in comparison
        rmse_fraction (float): fraction by which other RMSE should be better for drug to
                be considered as "better". If rmse_type is "relative", fraction should be greater than 1.,
                less than 1. otherwise.
        correlation_fraction (float, default None): fraction by which other correlation should be better for 
                drug to be considered as "better". Should be greater thatn 1. If None, this condidion is not applied.
        correlation_threshold (float): correlation that needs to be achieved in order
                for drug to be "better"
        
        Returns:
        None
        """
        # Compute DataFrame containing results from both Experiments
        total_df = self.merge_two_exp_results(other, flags, feature_name)
        
        # Extract data only for better performing drugs
        better_drugs = self.list_of_better_drugs(other, rmse_type, rmse_fraction, correlation_fraction,
                                                 correlation_threshold)
        return total_df[total_df["Drug ID"].isin(better_drugs)].sort_values("Drug ID")
        
    def barplots_of_performance_for_two_experiments(self, other, flags, feature_name, metric, 
                                             figsize=(35, 12), title_size=45, label_size=30, 
                                             tick_size=20, half=1, save_directory=None):
        """Plot barplots of a given performance metric for each drug. Two barplots for each drug 
        coming from two different Experiments.
        
        Arguments:
        other (Experiment): other considered Experiment object
        flags (list of strings): flags assigned to DataFrame entries from given object. First element 
                should be for this (self) instance
        feature_name (string): name of a column that differentiate between two results sets
        metric (string): which metric to plot
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        half (int): which half of the drugs to plot. If 0, all available drugs are plotted
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
        
        Returns:
        None
        """
        # Compute DataFrame containing results from both Experiments
        total_df = self.merge_two_exp_results(other, flags, feature_name)
        
        # Determine which half of the drugs to plot
        if half == 1:
            start_idx = 0
            end_idx = len(total_df["Drug ID"].unique()) // 2
        elif half == 2:
            start_idx = len(total_df["Drug ID"].unique()) // 2
            end_idx = len(total_df["Drug ID"].unique())
        else:
            start_idx = 0
            end_idx = len(total_df["Drug ID"].unique())
        drugs_to_plot = list(total_df["Drug ID"].unique())[start_idx:end_idx]
        
        # Actual plotting
        # Legend parameters
        params = {"legend.fontsize": 25,
                 "legend.handlelength": 2}
        plt.rcParams.update(params)
        
        fig = plt.figure(figsize=figsize)
        title = metric + " for each drug"
        plt.title(title, fontsize=45)
        plt.xlabel("Drug name", fontsize=30)
        plt.ylabel(metric, fontsize=30)
        sns.barplot("Drug Name", metric, hue=feature_name,
                    data=total_df[
                        total_df["Drug ID"].isin(drugs_to_plot)].sort_values(
                        metric, ascending=False))
        plt.xticks(rotation='vertical')
        plt.tick_params("both", labelsize=20)
        plt.legend(title = "")
        
        plt.gcf().subplots_adjust(bottom=0.25)

        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    
    def boxplots_of_performance_for_two_experiments(self, other, flags, feature_name, metric, 
                                                    title="Predictive performance",
                                                    figsize = (8, 6), title_size=25, 
                                                    label_size=20, tick_size=18, save_directory=None):
        """Create boxplots of a given performance metric for two Experiments, across
        all drugs.
        
        Arguments:
        other (Experiment): other considered Experiment object
        flags (list of strings): flags assigned to DataFrame entries from given object. First element 
                should be for this (self) instance
        feature_name (string): name of a column that differentiate between two results sets
        metric (string): which variable to plot, must be one of the column names in self.results_df
        title (string): title of the plot
        figsize (tuple): size of the figure
        title_size (int): font size of the title
        label_size (int): font size of axis labels
        tick_size (int): font size of axis ticks
        save_directory (string or None): directory to which save the plot. If None,
            plot is not saved
        
        Returns:
        None
        """
        # Compute DataFrame containing results from both Experiments
        total_df = self.merge_two_exp_results(other, flags, feature_name)
                  
        fig = plt.figure(figsize=figsize)
        plt.tick_params(labelsize=tick_size)
        plt.title(title, fontsize = title_size)
        plt.grid()

        sns.boxplot(x = feature_name, y = metric, data=total_df)
        plt.xlabel("", fontsize=label_size)
        plt.ylabel(metric, fontsize=label_size)
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    # Static methods
    @staticmethod
    def create_input_for_each_drug(drug_dict, drug_response_df, data_combination,
                                   gene_expression_df=None, cnv_binary_df=None,
                                   map_cl_id_and_feature_to_status=None,
                                   cell_lines_list_df=None, coding_variants_df=None,
                                   merck_signatures_df=None,
                                   feat_threshold=None, log=False,
                                   metric="AUC"):
        """Take in the dictionary with drug_id: Drug object pairs and create input data
        for each Drug.
        
        Arguments:
        drug_dict (dictionary): dictionary with drug_id: Drug object pairs
        drug_response_df (DataFrame): DataFrame of drug response data from GDSC
        gene_expression_df (DataFrame): original GDSC gene expression DataFrame
        cnv_binary_df (DataFrame): DataFrame from GDSC download tool with CNV data
        cell_line_list_df (DataFrame): cell line list from GDSC
        coding_variants_df (DataFrame): DataFrame with original mutation calls from GDSC
        data_combination (list of strings): determines which types of features to include, valid
                chocies are: "CNV", "mutation", "expression" and "tissue"
        feat_thresold (int): if not None, count number of Drugs with number of features greater
                than feat_threshold
        log (bool): provide or not the progress display
        
        Returns:
        None
        """
        s = 0
        c = 0
        sum_of_features = 0
        for ide in drug_dict:
            drug = drug_dict[ide]   # Current Drug object
            df = drug.create_full_data(drug_response_df, gene_expression_df=gene_expression_df, 
                                       cnv_binary_df=cnv_binary_df,
                                       map_cl_id_and_feature_to_status=map_cl_id_and_feature_to_status,
                                       cell_line_list=cell_lines_list_df,
                                       mutation_df=coding_variants_df,
                                       merck_signatures_df=merck_signatures_df,
                                       data_combination=data_combination,
                                       metric=metric
                                   )
            if feat_threshold:
                if df.shape[1] >= feat_threshold:
                    s += 1
            sum_of_features += (df.shape[1] - 14)
            c +=1
            if c % 10 == 0 and log:
                print(c, "drugs done")
        if feat_threshold:
            print("Number of drugs with number of features bigger than {}: {}".format(
            feat_threshold, s))
        if log:
            print("Mean number of features in {} drugs: {}".format(c, sum_of_features / c))
            
###############################################################################################
# Modeling class
###############################################################################################

class Modeling(object):
    """Basic class designed for performing machine learning experiments. This class alone doesn't involve
    any feature selection method. 
    """
    
    def __init__(self, name, param_grid, estimator_seeds, split_seeds, n_combinations=20, kfolds=5,
                             rforest_jobs=None, rforest_refit_jobs=None, max_iter=1000, tuning_jobs=None, 
                             scoring="neg_mean_squared_error", test_size=0.3):
        """ Modeling object initializer.
        
        Arguments:
            name (string): Name describing the experiment.
            param_grid (dict): Grid of parameters to search on during hyperparameter tuning.
            estimator_seeds (list of ints): Random States for predictive algorithms used.
            split_seeds (list of ints): Seeds for data splits.
            n_combinations (int): Number of parameters to try during hyperparameter tuning.
            kfolds (int): Number of folds of cross validation during hyperparameter tuning.
            rforest_jobs (int): Number of cores to use during fitting and predicting with 
                    Random Forest (during cross-validation hyperparamater tuning).
            rforest_refit_jobs (int): Number of cores to use during refitting RandomForest 
                    after cross-validation hyperparamater tuning.
            tuning_jobs (int): Number of cores to use during cross-validation.
            scoring (string): Function to optimize during parameter tuning.
            test_size (float): Fraction of whole data spent on test set.
            
        Returns:
            ModelingWithFeatureSelection object.
            """
        
        if len(estimator_seeds) != len(split_seeds):
            raise ValueError("Random forest seeds and split seeds must have the same length")
        self.name = name
        self.param_grid = param_grid
        self.estimator_seeds = estimator_seeds
        self.split_seeds = split_seeds
        self.n_combinations = n_combinations
        self.kfolds = kfolds
        self.rforest_jobs = rforest_jobs
        self.rforest_refit_jobs = rforest_refit_jobs
        self.max_iter = max_iter
        self.tuning_jobs = tuning_jobs
        self.scoring = scoring
        self.test_size = test_size
        
    def enet_fit_single_drug(self, X, y, tuning_seed=None,
                                        enet_seed=None, verbose=0, refit=True):
        """Perform single modeling given input data and labels, using Elastic Net regression.
        
        Modeling consists of hyperparameter tuning with kfold cross-validation and fitting 
        a model with best parameters on whole training data.
        
        Arguments:
            X (DataFrame): Training input data.
            y (Series): Training response variable.
            tuning_seed (int): Random State for parameter tuning.
            rforest_seed (int): Random State of Elastic Net Model.
            verbose (int): Controls verbosity of RandomizedSearchCV.
            
        Returns:
            grid: Fitted (if refit is True) RandomizedSearchCV object.
        """
        # Set elements of the pipeline, i.e. scaler and estimator
        scaler = StandardScaler()
        estimator = ElasticNet(random_state=enet_seed, max_iter=self.max_iter)
        
        # Create pipeline
        main_pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])    

        # Setup RandomizedSearchObject
        grid = model_selection.RandomizedSearchCV(main_pipeline, param_distributions=self.param_grid,
                                                 n_iter=self.n_combinations, scoring=self.scoring, 
                                                 cv=self.kfolds,
                                                 random_state=tuning_seed,
                                                 n_jobs=self.tuning_jobs,
                                                 verbose=verbose,
                                                 pre_dispatch='2*n_jobs',
                                                 refit=refit)
        # Fit the grid
        grid.fit(X, y)
        return grid
        
    def rf_fit_single_drug(self, X, y, tuning_seed=None,
                                        rforest_seed=None, verbose=0, refit=True):
        """Perform single modeling given input data and labels.
        
        Modeling consists of hyperparameter tuning with kfold cross-validation and fitting 
        a model with best parameters on whole training data.
        
        Arguments:
            X (DataFrame): Training input data.
            y (Series): Training response variable.
            tuning_seed (int): Random State for parameter tuning.
            rforest_seed (int): Random State of Random Forest Model.
            verbose (int): Controls verbosity of RandomizedSearchCV.
            
        Returns:
            grid: Fitted RandomizedSearchCV object.
        """
        # Set elements of the pipeline, i.e. scaler and estimator
        scaler = StandardScaler()
        estimator = RandomForestRegressor(random_state=rforest_seed, n_jobs=self.rforest_jobs)
        
        # Create pipeline
        main_pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])    

        # Setup RandomizedSearchObject
        grid = model_selection.RandomizedSearchCV(main_pipeline, param_distributions=self.param_grid,
                                                 n_iter=self.n_combinations, scoring=self.scoring, 
                                                 cv=self.kfolds,
                                                 random_state=tuning_seed,
                                                 n_jobs=self.tuning_jobs,
                                                 verbose=verbose,
                                                 pre_dispatch='2*n_jobs',
                                                 refit=refit)
        # Fit the grid
        grid.fit(X, y)
        return grid
    
    def evaluate_single_drug(self, grid, X, y):
        """Evaluate provided fitted (or not) model on a test data.
        
        Arguments:
            grid (RandomizedSearchCV object): Previously fitted best model.
            X (DataFrame): Test input data.
            y (Series): Test response variable.
            
        Returns:
            model_test_scores (namedtuple): Results on a test set.
            grid.cv_results_ (dict): Full results of parameter tuning with cross-validation.
            grid.best_params_ (dict): Parameters of best found model.
        """
        if grid.refit:
            pred = grid.predict(X)
            # Record results as named tuple
            ModelTestScores = collections.namedtuple("ModelTestScores", ["cv_best_RMSE", "test_RMSE",
                                                            "test_explained_variance", "test_correlation"])
            model_test_scores = ModelTestScores((-grid.best_score_) ** 0.5, 
                                                metrics.mean_squared_error(y, pred) ** 0.5,
                                               metrics.explained_variance_score(y, pred), 
                                                pearsonr(y, pred))

            return model_test_scores, grid.cv_results_, grid.best_params_
        
        # If grid is not refitted, we just want the CV results
        else:   
            return (-grid.best_score_) ** 0.5, grid.cv_results_, grid.best_params_
        
    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, tuning_seed=None,
                        rforest_seed=None, verbose=0, test_set=True):
        """Perform fitting and evaluation model in a single method (using Random Forest). Modeling 
        consists of hyperparameter tuning with cross-validation and fitiing the model on whole 
        training data. Evaluation is done either on separate test set or on cross-validation.
        
        Arguments:
            X_train (array): Training input data.
            y_train (Series): Training labels.
            X_test (array): Test input data.
            y_test (Series): Test labels.
            tuning_seed (int): Random State of RandomizedSearch object.
            rforest_seed (int): Random State of RandomForestRegressor.
            verbose (int): Controls the verbosity of RandomizedSearch objects.
            test_set (bool): Whether or not evaluate on a separate test set.
            
        Returns:
            (if test_set):
                model_test_scores (namedtuple): Results on a test set.
                grid.cv_results_ (dict): Full results of parameter tuning with cross-validation.
                grid.best_params_ (dict): Parameters of best found model.
            (else):
                grid.best_score (namedtuple): Best result obtained during CV.
                grid.cv_results_ (dict): Full results of parameter tuning with cross-validation.
                grid.best_params_ (dict): Parameters of best found model.
        """
            
        # Set elements of the pipeline, i.e. scaler and estimator
        scaler = StandardScaler()
        estimator = RandomForestRegressor(random_state=rforest_seed, n_jobs=self.rforest_jobs)
        
        # Create pipeline
        main_pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])    

        # Setup RandomizedSearchObject
        grid = model_selection.RandomizedSearchCV(main_pipeline, param_distributions=self.param_grid,
                                                 n_iter=self.n_combinations, scoring=self.scoring, 
                                                  cv=self.kfolds,
                                                 random_state=tuning_seed,
                                                 n_jobs=self.tuning_jobs,
                                                 verbose=verbose,
                                                 pre_dispatch='2*n_jobs',
                                                 refit=False)
        # Fit the grid
        grid.fit(X_train, y_train)
        
        # Parse best params to pipeline
        main_pipeline.set_params(**grid.best_params_)
        main_pipeline.named_steps["estimator"].n_jobs = self.rforest_refit_jobs
        
        if test_set:
            # Refit on whole training data
            main_pipeline.fit(X_train, y_train)
            pred = grid.predict(X_test)
            # Record results as named tuple
            ModelTestScores = collections.namedtuple("ModelTestScores", ["cv_best_RMSE", "test_RMSE",
                                                            "test_explained_variance", "test_correlation"])
            model_test_scores = ModelTestScores((-grid.best_score_) ** 0.5, 
                                                metrics.mean_squared_error(y_test, pred) ** 0.5,
                                               metrics.explained_variance_score(y_test, pred), 
                                                pearsonr(y_test, pred))

            return model_test_scores, grid.cv_results_, grid.best_params_
        
        else:
            return (-grid.best_score_) ** 0.5, grid.cv_results_, grid.best_params_
        
        
        
    
    def evaluate_dummy_model_single_split(self, X_train, X_test, y_train, y_test):
        """Fit and evaluate the performance of dummy model, only on the test set."""
        # Set up and fit Dummy Regressor
        dummy = DummyRegressor()
        dummy.fit(X_train, y_train)
        
        # Get dummy predictions on the test set
        dummy_preds = dummy.predict(X_test)
        
        # Performance of dummy model as namedtuple
        DummyScores = collections.namedtuple("DummyScores", ["test_RMSE", "test_explained_variance",
                                                "test_correlation"])
        dummy_performance = DummyScores(metrics.mean_squared_error(y_test, dummy_preds) ** 0.5,
                                       metrics.explained_variance_score(y_test, dummy_preds),
                                       pearsonr(y_test, dummy_preds))
        return dummy_performance
    
    def enet_model_over_data_splits(self, X, y, verbose=0, log=False):
        """Perform full modeling over data splits, using ElasticNet. Modeling involves hyperparameter 
        tuning with cross-validation, training on whole training data, and evaluating on the test set.
        This process is repeated for few data splits (random states of data splits are contained
        in self.split_seeds). 
        
        Arguments:
            X (array): Input data (whole, not just the test set).
            y (Series): Labels.
            verbose (int): Controls the verbosity of RandomizedSearch object.
            log (bool): Controls information display.
            
        Returns:
            results_for_splits (dict): Dictionary with data split seeds as keys and namedtuples
                    with results as values.
            cv_results_for_splits (dict): Dictionary with data split seeds as keys and full 
                    results of CV as values.
            best_parameters_for_splits (dict): Dictionary with data split seeds as keys and
                    parameters of best found model as values.
            dummy_for_splits (dict): Dictionary with data split seeds as keys and results
                    of dummy model as values.
            tuning_seeds_for_splits (dict): Dictionary with data split seeds as keys and random
                    states of RandomizedSearch objects as values.
        """
        # Initialize dictionary for storage of test results for given split
        results_for_splits = {}
        # Initialize dictionary with full results of cross-validation
        cv_results_for_splits = {}
        # Initialize dictonary with best parameters
        best_parameters_for_splits = {}
        # Initialize dictionary with dummy performance
        dummy_for_splits = {}
        # Initialize dictionary for tuning seed storage
        tuning_seeds_for_splits = {}
        
        # Initialize list of tuning seeds
        tuning_seeds = np.random.randint(0, 101, size=len(self.split_seeds))
        
        # Iterate over split seeds
        for i in range(len(self.split_seeds)):
            if log:
                print("Modeling for {} out of {} data splits".format(i + 1, len(self.split_seeds)))
                print()
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Record tuning seed
            tuning_seeds_for_splits[self.split_seeds[i]] = tuning_seeds[i]
            
            # Evaluate dummy
            dummy_for_splits[self.split_seeds[i]] = self.evaluate_dummy_model_single_split(
            X_train, X_test, y_train, y_test)
            
            # Fit the model
            grid = self.enet_fit_single_drug(X_train, y_train, tuning_seed=tuning_seeds[i],
                                          enet_seed=self.estimator_seeds[i], verbose=verbose)
            
            # Evaluate the model
            model_test_scores, cv_results, best_parameters = self.evaluate_single_drug(grid, X_test, y_test)
            
            # Record the results
            results_for_splits[self.split_seeds[i]] = model_test_scores
            cv_results_for_splits[self.split_seeds[i]] = cv_results
            best_parameters_for_splits[self.split_seeds[i]] = best_parameters
            
        if log:
            print("Modeling done for all splits")
        
        return (results_for_splits, cv_results_for_splits, best_parameters_for_splits, 
                dummy_for_splits, tuning_seeds_for_splits)
    
    def rf_model_over_data_splits(self, X, y, verbose=0, log=False):
        """Perform full modeling over data splits, using RandomForest. Modeling involves hyperparameter 
        tuning with cross-validation, training on whole training data, and evaluating on the test set.
        This process is repeated for few data splits (random states of data splits are contained
        in self.split_seeds). 
        
        Arguments:
            X (array): Input data (whole, not just the test set).
            y (Series): Labels.
            verbose (int): Controls the verbosity of RandomizedSearch object.
            log (bool): Controls information display.
            
        Returns:
            results_for_splits (dict): Dictionary with data split seeds as keys and namedtuples
                    with results as values.
            cv_results_for_splits (dict): Dictionary with data split seeds as keys and full 
                    results of CV as values.
            best_parameters_for_splits (dict): Dictionary with data split seeds as keys and
                    parameters of best found model as values.
            dummy_for_splits (dict): Dictionary with data split seeds as keys and results
                    of dummy model as values.
            tuning_seeds_for_splits (dict): Dictionary with data split seeds as keys and random
                    states of RandomizedSearch objects as values.
        """
        # Initialize dictionary for storage of test results for given split
        results_for_splits = {}
        # Initialize dictionary with full results of cross-validation
        cv_results_for_splits = {}
        # Initialize dictonary with best parameters
        best_parameters_for_splits = {}
        # Initialize dictionary with dummy performance
        dummy_for_splits = {}
        # Initialize dictionary for tuning seed storage
        tuning_seeds_for_splits = {}
        
        # Initialize list of tuning seeds
        tuning_seeds = np.random.randint(0, 101, size=len(self.split_seeds))
        
        # Iterate over split seeds
        for i in range(len(self.split_seeds)):
            if log:
                print("Modeling for {} out of {} data splits".format(i + 1, len(self.split_seeds)))
                print()
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Record tuning seed
            tuning_seeds_for_splits[self.split_seeds[i]] = tuning_seeds[i]
            
            # Evaluate dummy
            dummy_for_splits[self.split_seeds[i]] = self.evaluate_dummy_model_single_split(
            X_train, X_test, y_train, y_test)
            
            # Fit the model
            grid = self.rf_fit_single_drug(X_train, y_train, tuning_seed=tuning_seeds[i],
                                          rforest_seed=self.estimator_seeds[i], verbose=verbose)
            
            # Evaluate the model
            model_test_scores, cv_results, best_parameters = self.evaluate_single_drug(grid, X_test, y_test)
            
            # Record the results
            results_for_splits[self.split_seeds[i]] = model_test_scores
            cv_results_for_splits[self.split_seeds[i]] = cv_results
            best_parameters_for_splits[self.split_seeds[i]] = best_parameters
            
        if log:
            print("Modeling done for all splits")
        
        return (results_for_splits, cv_results_for_splits, best_parameters_for_splits, 
                dummy_for_splits, tuning_seeds_for_splits)
        
    
    @staticmethod
    def selectKImportance(X, sorted_importance_indices, k):
        """Reduce input data to k important features.
        
        Arguments:
            X (array): Input data.
            sorted_importance_indices (1D array): Array with sorted indices corresponding to 
                    features, sorting is based on importance of features.
            k (int): Number of features to choose.
        """
        return X.iloc[:,sorted_importance_indices[:k]]
    
###############################################################################################
# ModelingWithFeatureSelection class
###############################################################################################

class ModelingWithFeatureSelection(Modeling):
    """Class designed to perform ML modeling with feature selection methods, inherits from Modeling.
    """
    def __init__(self, name, param_grid, estimator_seeds, split_seeds, n_combinations=20, 
                            kfolds=5, n_combinations_importances=30, kfolds_importances=10, 
                            rforest_jobs=None, rforest_refit_jobs=None, tuning_jobs=None,
                            ss_lambda_grid=[0.0001, 0.001, 0.01], ss_n_bootstrap_iterations=100,
                            ss_threshold=0.6, ss_n_jobs=1, max_iter=1000,
                            scoring="neg_mean_squared_error", test_size=0.3):
        """ ModelingWithFeatureSelection initializer.

        Arguments:
            name (string): Name describing the experiment.
            param_grid (dict): Grid of parameters to search on during hyperparameter tuning.
            rforest_seeds (list of ints): Random States for predictive algorithms used.
            split_seeds (list of ints): Seeds for data splits.
            n_combinations (int): Number of parameters to try during hyperparameter tuning.
            kfolds (int): Number of folds of cross validation during hyperparameter tuning.
            n_combinations_importances (int): Number of parameters to try during extracting the
            feature importances.
            kfolds_importances (int): Number of folds of cross validation during extracting the
            feature importances.
            rforest_jobs (int): Number of cores to use during fitting and predicting with 
            Random Forest.
            rforest_refit_jobs (int): Number of cores to use during refitting RandomForest 
            after cross-validation hyperparamater tuning.
            tuning_jobs (int): Number of cores to use during cross-validation.
            ss_lambda_grid (dict): Lambdas to iterate over for StabilitySelection.
            ss_n_bootstrap_iterations (int): Number of iterations for StabilitySelection.
            ss_threshold (float): Threshold to use for features stability scores.
            ss_n_jobs (int): Number of cores to use for StabilitySelection.
            scoring (string): Function to optimize during parameter tuning.
            test_size (float): Fraction of whole data spent on test set.

        Returns:
            ModelingWithFeatureSelection object.
        """
        # Call parent class initializer
        super().__init__(name, param_grid, estimator_seeds, split_seeds, n_combinations, kfolds,
        rforest_jobs, rforest_refit_jobs, max_iter, tuning_jobs, scoring, test_size)

        # Random Forest specific attributes
        self.n_combinations_importances = n_combinations_importances
        self.kfolds_importances = kfolds_importances

        # ElasticNet / StabilitySelection specific attributes
        self.ss_lambda_grid = ss_lambda_grid
        self.ss_n_bootstrap_iterations = ss_n_bootstrap_iterations
        self.ss_threshold = ss_threshold
        self.ss_n_jobs = ss_n_jobs
        
    def enet_feat_importances_single_drug(self, X, y, lasso_seed=None, ss_seed=None, verbose=0):
        """Extract feature importances using StabilitySelection with ElasticNet.

        Arguments:
            X (DataFrame): Training input data.
            y (Series): Training response variable.
            lasso_seed (int): Random State of Lasso model.
            ss_seed (int): Random State for StabilitySelection.
            verbose (int): Controls verbosity of StabilitySelection.

        Returns:
            selector: Fitted StabilitySelection object.
        """
        # Set up elements of modeling pipeline, i.e. scaler and estimator
        scaler = StandardScaler()
        estimator = Lasso(random_state=lasso_seed, max_iter=self.max_iter)

        # Create the pipeline
        pipeline = Pipeline([
        ("scaler", scaler),
        ("estimator", estimator)
        ])

        # Setup StabilitySelection object
        selector = StabilitySelection(base_estimator=pipeline, 
                            lambda_name="estimator__alpha", 
                            lambda_grid=self.ss_lambda_grid,
                            n_bootstrap_iterations=self.ss_n_bootstrap_iterations,
                            threshold=self.ss_threshold,
                            verbose=verbose,
                            n_jobs=self.ss_n_jobs,
                            random_state=ss_seed)

        selector.fit(X, y)
        return selector
    
    def enet_fit_grid_range_of_feats(self, X_train, y_train,
                    selector, stability_scores_thresholds, 
                    tuning_seed=None,
                    enet_seed=None, verbose=0, log=False):
        """Perform modeling with ElasticNet for different feature numbers.

        Iterate over thresholds of stability scores. For every threshold, perform hyperparameter
        tuning and evaluate using same training data (using cross-validation score).

        Arguments:
            X_train (DataFrame): Training input data.
            y_train (Series): Training response variable.
            selector (StabilitySelection): Fitted StabilitySelection object.
            stability_scores_thresholds (list of floats): Stability scores determining 
                    which features to include.
            tuning_seed (int): Random State for parameter tuning.
            enet_seed (int): Random State of ElasticNet model.
            verbose (int): Controls verbosity of RandomizedSearchCV.
            log (bool): Controls output during running.

        Returns:
            results (dict): Dictionary with feature numbers k as keys and CV results as values.
            best_parameters (dict): Dictionary with feature numbers k as keys and parameters
                    of best found model as values.
        """
        # Initialize dictionary for storage of results
        results = {}
        # Initialize dictionary for storage of best parameters
        best_parameters = {}
        
        # Iterate over stability thresholds
        for stability_thresh in stability_scores_thresholds:
            # Extract corresponding data with reduced features
            X_train_k_feats = selector.transform(X_train, threshold=stability_thresh)
            k = X_train_k_feats.shape[1]
            
            if k > 0:  # If k is smaller, do not perform modeling at all
                # Train and evaluate during cross-validation
                grid = self.enet_fit_single_drug(X_train_k_feats, y_train, tuning_seed=tuning_seed,
                                    enet_seed=enet_seed, verbose=verbose, refit=False)
                
                cv_best_score, cv_full_results, grid_best_params = self.evaluate_single_drug(
                                    grid, X_train, y_train)

                # Record the results for k features and threshold
                results[(k, stability_thresh)] = cv_best_score
                best_parameters[(k, stability_thresh)] = grid_best_params
                
                if log:
                    print("Done modeling with threshold {} and {} features".format(
                                    stability_thresh, k))
        return results, best_parameters
    
    def enet_get_best_k(self, X, y, stability_scores_thresholds,
                                  verbose_importances=0, 
                                  verbose_kfeats=0, log=False):
                                
        """Perform modeling over range of features k for few data splits in order to obtain best
        feature number k and stability threshold for particular drug.
        
        Arguments:
            X (DataFrame): Whole input data.
            y (Series): Whole response variable.
            stability_scores_thresholds (list of floats): Stability scores determining 
                    which features to include.
            verbose_importances (int): Controls verbosity of RandomizedSearchCV during feature
                    importances extraction.
            verbose_kfeats (int): Controls verbosity of RandomizedSearchCV during modeling with
                    different numbers of features.
            log (bool): Controls output during running.
        
        Returns:
            results_for_splits (dict): Nested dictionary with split seeds as keys, and dictionaries
                    with feature numbers as keys and results as values.
            best_parameters_for_splits (dict): Dictionary with split seeds as keys and dictionaries
                    with feature numbers as keys and best parameters as values.
            selectors_for_splits (dict): Dictionary with split seeds as keys and fitted
                    StabilitySelection objects as values.
            tuning_seeds_for_splits (dict): Dictionary with split seeds as keys and random states
                    of RandomizedSearch as values.
        
        """
        # Initialize dictionary for results over split seeds s and feature numbers k
        results_for_splits = {}
        # Initialize dictionary with selectors per split
        selectors_for_splits = {}
        # Initialize dictionary with best parameters over s and k
        best_parameters_for_splits = {}
        # Initialize dictionary for tuning seed per data split
        tuning_seeds_for_splits = {}
        # Initialize list of tuning seeds
        tuning_seeds = np.random.randint(0, 101, size=len(self.split_seeds))
        
        # Iterate over data splits
        for i in range(len(self.split_seeds)):
            if log:
                print()
                print("Modeling for {} out of {} data splits".format(i + 1, len(self.split_seeds)))
                print()
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                            X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Fit selector in order to get feature importances
            selector = self.enet_feat_importances_single_drug(
                                        X_train, y_train, 
                                        lasso_seed=self.estimator_seeds[i], 
                                        ss_seed=self.estimator_seeds[i], 
                                        verbose=verbose_importances)
            
            # Record tuning seed
            tuning_seeds_for_splits[self.split_seeds[i]] = tuning_seeds[i]
            # Record selector
            selectors_for_splits[self.split_seeds[i]] = selector
            
            # Perform modeling and evaluation on range of features
            results_range_of_feats, best_parameters_for_ks = self.enet_fit_grid_range_of_feats(
                                    X_train, y_train,
                                    selector,
                                    stability_scores_thresholds,
                                    tuning_seed=tuning_seeds[i],
                                    enet_seed=self.estimator_seeds[i],
                                    verbose=verbose_kfeats,
                                    log=log)
            
            # Perform modeling for all features
            grid = self.enet_fit_single_drug(X_train, y_train, tuning_seed=tuning_seeds[i],
                                    enet_seed=self.estimator_seeds[i], verbose=verbose_kfeats, refit=False)
            
            cv_best_score, cv_full_results, grid_best_params = self.evaluate_single_drug(
                                grid, X_train, y_train)
            k = X_train.shape[1]
            if log:
                print("Done modeling with threshold {} and {} features".format(0.0, k))
                
            # Add results for all features into dictionary
            results_range_of_feats[(k, 0.0)] = cv_best_score
            best_parameters_for_ks[(k, 0.0)] = grid_best_params
            
            # Record best parameters for ks
            best_parameters_for_splits[self.split_seeds[i]] = best_parameters_for_ks
            # Record results for this split
            results_for_splits[self.split_seeds[i]] = results_range_of_feats
            
        if log:
            print("Modeling done for all splits")
            
        return (results_for_splits, best_parameters_for_splits, 
                selectors_for_splits, tuning_seeds_for_splits)
    
    def enet_model_over_data_splits(self, X, y, stability_scores_thresholds,
                                  verbose_importances=0, 
                                  verbose_kfeats=0, log=False):
        """Perform full modeling over data splits. For a single data split, modeling consist of:
                1) Extraction of feature importances using whole training data (via
                        StabilitySelection selector).
                2) Hyperparamater tuning and evaluation using just training data
                   for range of feature importance scores.
       After that, best threshold is determined. Then, for the same data splits, model is trained
       on whole training set with best found parameters and thresh. After that, model is evaluated 
       on the test set.
       
       Arguments:
           X (DataFrame): Whole input data.
           y (Series): Whole response variable.
           stability_scores_thresholds (list of floats): Stability scores determining 
                    which features to include.
           verbose_importances (int): Controls verbosity of RandomizedSearchCV during feature
                importances extraction.
           verbose_kfeats (int): Controls verbosity of RandomizedSearchCV during modeling with
                different numbers of features.
           log (bool): Controls output during running.
        
        Returns:
            test_results_for_splits (dict): Dictionary with split seeds as keys and final results on
                    a test set (in form of namedtuple) as values.
            dummy_for_splits (dict): Dictionary with split seeds as keys and results of dummy model
                    as values.
            selectors_for_splits (dict): Dictionary with split seeds as keys and fitted
                    StabilitySelection objects as values.
            best_parameters_for_splits (dict): Dictionary with split seeds as keys and dictionaries
                    with feature numbers as keys and best parameters as values.
            tuning_seeds_for_splits (dict): Dictionary with split seeds as keys and random states
                    of RandomizedSearch as values.
            best_k (int): Best found feature number k.
        """
        # Initialize dictionary for storage of test results for given split
        test_results_for_splits = {}
        # Initialize dictionary with dummy performance
        dummy_for_splits = {}
        # Get results across splits s and feature numbers k
        (results_for_splits, best_parameters_for_splits, 
        selectors_for_splits, tuning_seeds_for_splits) = self.enet_get_best_k(X, y, stability_scores_thresholds,
                                                          verbose_importances, 
                                                          verbose_kfeats, log)
        
        # Determine the best threshold
        # First, create a new result dictionary with only stability thresholds as keys
        new_results_for_splits = {}
        for s in results_for_splits:
            results_over_feats = {}
            for tup in results_for_splits[s]:
                results_over_feats[tup[1]] = results_for_splits[s][tup]
            new_results_for_splits[s] = results_over_feats
            
        # Then, average results for particular threshold over data splits
        average_results_per_thresh = []
        for thresh in stability_scores_thresholds:
            # Thresholds has to be included for all splits if to be considered
            present_in_all = True
            for s in self.split_seeds:
                if thresh not in new_results_for_splits[s]:
                    present_in_all = False
                    break
            if present_in_all:
                sum_of_results = 0
                for s in new_results_for_splits.keys():
                    sum_of_results += new_results_for_splits[s][thresh]
                average_result = sum_of_results / len(new_results_for_splits)
                average_results_per_thresh.append((thresh, average_result))
            
        # Get the best threshold
        best_threshold = min(average_results_per_thresh, key=lambda x: x[1])[0]
        
        # Perform the modeling for best found threshold
        # Iterate over data splits (again, same splits as during finding best k)
        for i in range(len(results_for_splits)):
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Evaluate dummy
            dummy_for_splits[self.split_seeds[i]] = self.evaluate_dummy_model_single_split(
            X_train, X_test, y_train, y_test)
            
            
            # Get data for best threshold
            # First, get appropriate selector
            sel = selectors_for_splits[self.split_seeds[i]]
            # Extract corresponding data
            X_train_kfeats = sel.transform(X_train, threshold=best_threshold)
            X_test_kfeats = sel.transform(X_test, threshold=best_threshold)
            
            # No need for parameter tuning because we have best parameters
            # Set elements of the pipeline, i.e. scaler and estimator
            scaler = StandardScaler()
            estimator = ElasticNet(random_state=self.estimator_seeds[i], max_iter=self.max_iter)

            # Create pipeline
            main_pipeline = Pipeline([
                ("scaler", scaler),
                ("estimator", estimator)
            ])    
            
            # No need for parameter tuning because we have best parameters
            # Find best parameters
            for tup in best_parameters_for_splits[self.split_seeds[i]]:
                if tup[1] == best_threshold:
                    best_tup = tup
                    break
            # Fill best parameters to the to the estimator
            main_pipeline.set_params(**best_parameters_for_splits[self.split_seeds[i]][best_tup])
            
            # Fit model on whole training data
            main_pipeline.fit(X_train_kfeats, y_train)
            
            # Evaluate the model and record the results
            pred = main_pipeline.predict(X_test_kfeats)
            # Record results in corresponding Experiment fields, mostly as named tuples
            # Classification performance
            ModelTestScores = collections.namedtuple("ModelTestScores", ["test_RMSE",
                                                            "test_explained_variance", "test_correlation"])
            model_test_scores = ModelTestScores(metrics.mean_squared_error(y_test, pred) ** 0.5,
                                               metrics.explained_variance_score(y_test, pred), 
                                                pearsonr(y_test, pred))
            # Record the best results
            test_results_for_splits[self.split_seeds[i]] = model_test_scores
            
        return (test_results_for_splits, dummy_for_splits, selectors_for_splits, 
                best_parameters_for_splits, 
                tuning_seeds_for_splits, best_threshold)
       
        
        
    def rf_feat_importances_single_drug(self, X, y, rforest_seed=None, tuning_seed=None, verbose=0):
        """Extract feature importances using Random Forest.
        
        First, hyperparameter tuning is performed using kfold cross-validation (sklearn's RandomizedSearchCV). 
        Then, model is trained on whole training data and feature importances are extracted. In order to use
        different number of cores during refitting the model, it is done separately manually, "refit" argument 
        of RandomizedSearch is set to False.
        
        Arguments:
            X (DataFrame): Trainig input data.
            y (Series): Training response variable.
            rforest_seed (int): Random State of Random Forest Model.
            tuning_seed (int): Random State for parameter tuning.
            verbose (int): Controls verbosity of RandomizedSearchCV.
            
        Returns:
            grid: Fitted RandomizedSearchCV object.
            importances: List of tuples (feature name, importance coefficient).
        """
        # Setup elements of the pipeline; scaler and estimator
        scaler = StandardScaler()
        estimator = RandomForestRegressor(random_state=rforest_seed, 
                                          n_jobs=self.rforest_jobs)
        
        # Create pipeline
        main_pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])    

        # Setup RandomizedSearch
        grid = model_selection.RandomizedSearchCV(main_pipeline,
                                                  param_distributions=self.param_grid,
                                                  n_iter=self.n_combinations_importances, 
                                                  scoring=self.scoring,
                                                  cv=self.kfolds_importances,
                                                  random_state=tuning_seed,
                                                  verbose=verbose,
                                                  n_jobs=self.tuning_jobs,
                                                  pre_dispatch='2*n_jobs',
                                                  refit=False)

        # Fit the grid
        grid.fit(X, y)
        
        # Parse best params to pipeline
        main_pipeline.set_params(**grid.best_params_)
        # Set the number of cores during refitting
        main_pipeline.named_steps["estimator"].n_jobs = self.rforest_refit_jobs
        
        # Refit the model
        main_pipeline.fit(X, y)
        
        # Get the feature importances along with feature names
        clf = main_pipeline.named_steps["estimator"]
        importances = [x for x in zip(X.columns, clf.feature_importances_)]
        return grid, importances
    
    def rf_fit_grid_range_of_feats(self, X_train, y_train,
                                        sorted_importance_indices,
                                        feature_numbers, tuning_seed=None,
                                        rforest_seed=None, verbose=0, log=False):
        """Perform modeling for a given range of number of input features.
        
        Iterate over numbers of features k. For every k, perform parameter tuning and evaluate 
        using cross-validation on training data.
        
        Arguments:
            X_train (DataFrame): Training input data.
            X_test (DataFrame): Test input data.
            y_train (Series): Training response variable.
            y_test (Series): Test response variable.
            sorted_importance_indices (array): Sorted indexes corresponding to specific features.
            feature_numbers (list of ints): Feature numbers k for which do the modeling.
            tuning_seed (int): Random State for parameter tuning.
            rforest_seed (int): Random State of Random Forest Model.
            verbose (int): Controls verbosity of RandomizedSearchCV.
            log (bool): Controls output during running.
            
        Returns:
            results (dict): Dictionary with feature numbers k as keys and CV results as values.
            best_parameters (dict): Dictionary with feature numbers k as keys and parameters
                    of best found model as values.
        """
        # Initialize dictionary for storage of results
        results = {}
        # Initialize dictionary for storage of best parameters
        best_parameters = {}
        # Iterate over number of features
        for k in feature_numbers:
            X_train_kfeats = self.selectKImportance(X_train, sorted_importance_indices, k)
            
            # Train and evaluate
            cv_performance, _, best_params = self.fit_and_evaluate(X_train_kfeats, y_train, X_train_kfeats, y_train, 
                                                                   tuning_seed=tuning_seed, rforest_seed=rforest_seed, 
                                                                   verbose=verbose, test_set=False)
            
            # Record the results for k features
            results[k] = cv_performance
            best_parameters[k] = best_params
            if log:
                print("Done modeling with {} features".format(k))
            
        return results, best_parameters
    
    def rf_get_best_k(self, X, y, feature_numbers,
                                  verbose_importances=0, 
                                  verbose_kfeats=0, log=False):
                                
        """Perform modeling over range of features k for few data splits in order to obtain best
        feature number k for particular drug.
        
        Arguments:
            X (DataFrame): Whole input data.
            y (Series): Whole response variable.
            feature_numbers (list of ints): Feature numbers k for which do the modeling.
            verbose_importances (int): Controls verbosity of RandomizedSearchCV during feature
                    importances extraction.
            verbose_kfeats (int): Controls verbosity of RandomizedSearchCV during modeling with
                    different numbers of features.
            log (bool): Controls output during running.
        
        Returns:
            results_for_splits (dict): Nested dictionary with split seeds as keys, and dictionaries
                    with feature numbers as keys and results as values.
            best_parameters_for_splits (dict): Dictionary with split seeds as keys and dictionaries
                    with feature numbers as keys and best parameters as values.
            importances_for_splits (dict): Dictionary with split seeds as keys and lists of feature 
                    importances as values.
            tuning_seeds_for_splits (dict): Dictionary with split seeds as keys and random states
                    of RandomizedSearch as values.
        
        """
        # Initialize dictionary for results over split seeds s and feature numbers k
        results_for_splits = {}
        # Initialize dictionary with feature importances per data split
        importances_for_splits = {}
        # Initialize dictionary with best parameters over s and k
        best_parameters_for_splits = {}
        # Initialize dictionary for tuning seed per data split
        tuning_seeds_for_splits = {}
        # Initialize list of tuning seeds
        tuning_seeds = np.random.randint(0, 101, size=len(self.split_seeds))
        
        # Iterate over data splits
        for i in range(len(self.split_seeds)):
            if log:
                print("Modeling for {} out of {} data splits".format(i + 1, len(self.split_seeds)))
                print()
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Get vector with feature importances
            grid, feat_importances = self.rf_feat_importances_single_drug(X_train, y_train, 
                                        tuning_seed=tuning_seeds[i],
                                        rforest_seed=self.estimator_seeds[i], verbose=verbose_importances)
            # Record tuning seed
            tuning_seeds_for_splits[self.split_seeds[i]] = tuning_seeds[i]
            # Record feature importances
            importances_for_splits[self.split_seeds[i]] = feat_importances
            # Extract just the importance coeficients
            importances = np.array([x[1] for x in feat_importances])
            # Evaluate performance for m (all features)
            result_m_features, _, _ = self.evaluate_single_drug(grid, X_test, y_test)
            
            # Perform modeling and evaluation on range of features
            sorted_importance_indices = importances.argsort()[::-1]
            results_range_of_feats, best_parameters_for_ks = self.rf_fit_grid_range_of_feats(X_train, y_train,
                                                                        sorted_importance_indices,
                                                                        feature_numbers,
                                                                        tuning_seed=tuning_seeds[i],
                                                                        rforest_seed=self.estimator_seeds[i],
                                                                        verbose=verbose_kfeats,
                                                                        log=log)
            # Record best parameters for ks
            best_parameters_for_splits[self.split_seeds[i]] = best_parameters_for_ks
            # Add results for all features
            results_range_of_feats[len(feat_importances)] = result_m_features
            
            # Record results for this split
            results_for_splits[self.split_seeds[i]] = results_range_of_feats
            
        if log:
            print("Modeling done for all splits")
            
        return (results_for_splits, best_parameters_for_splits, 
                importances_for_splits, tuning_seeds_for_splits)
    
    
    def rf_model_over_data_splits(self, X, y, feature_numbers,
                                  verbose_importances=0, 
                                  verbose_kfeats=0, log=False):
        """Perform full modeling over data splits. For a single data split, modeling consist of:
                1) Extraction of feature importances using whole training data.
                2) Hyperparamater tuning and evaluation using just training data
                   for range of feature numbers k.
       After that, best k is determined. Then, for the same data splits model is trained
       on whole training set with best found parameters and k. After that, model is evaluated 
       on the test set.
       
       Arguments:
           X (DataFrame): Whole input data.
           y (Series): Whole response variable.
           feature_numbers (list of ints): Feature numbers k for which do the modeling.
           verbose_importances (int): Controls verbosity of RandomizedSearchCV during feature
                importances extraction.
           verbose_kfeats (int): Controls verbosity of RandomizedSearchCV during modeling with
                different numbers of features.
           log (bool): Controls output during running.
        
        Returns:
            test_results_for_splits (dict): Dictionary with split seeds as keys and final results on
                    a test set (in form of namedtuple) as values.
            dummy_for_splits (dict): Dictionary with split seeds as keys and results of dummy model
                    as values.
            importances_for_splits (dict): Dictionary with split seeds as keys and feature importances
                    as values.
            best_parameters_for_splits (dict): Dictionary with split seeds as keys and dictionaries
                    with feature numbers as keys and best parameters as values.
            tuning_seeds_for_splits (dict): Dictionary with split seeds as keys and random states
                    of RandomizedSearch as values.
            best_k (int): Best found feature number k.
        """
        # Initialize dictionary for storage of test results for given split
        test_results_for_splits = {}
        # Initialize dictionary with dummy performance
        dummy_for_splits = {}
        # Get results across splits s and feature numbers k
        (results_for_splits, best_parameters_for_splits, 
        importances_for_splits, tuning_seeds_for_splits) = self.rf_get_best_k(X, y, feature_numbers,
                                                          verbose_importances, 
                                                          verbose_kfeats, log)
        
        # Determine the best k
        # First, average results for particular k over data splits
        average_results_per_k = []
        for k in feature_numbers:
            sum_of_results = 0
            for s in results_for_splits.keys():
                sum_of_results += results_for_splits[s][k]
            average_result = sum_of_results / len(results_for_splits)
            average_results_per_k.append((k, average_result))
        # Get the maximum k
        best_k = min(average_results_per_k, key=lambda x: x[1])[0]
        
        # Perform the modeling for best found k
        # Iterate over data splits (again, same splits as during finding best k)
        for i in range(len(results_for_splits)):
            # Split data into training and test set
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=self.test_size, random_state=self.split_seeds[i])
            
            # Evaluate dummy
            dummy_for_splits[self.split_seeds[i]] = self.evaluate_dummy_model_single_split(
            X_train, X_test, y_train, y_test)
            
            
            # Get data for best_k features
            importances = np.array([x[1] for x in importances_for_splits[self.split_seeds[i]]])
            sorted_importance_indices = importances.argsort()[::-1]
            X_train_kfeats = self.selectKImportance(X_train, sorted_importance_indices, best_k)
            X_test_kfeats = self.selectKImportance(X_test, sorted_importance_indices, best_k)
            
            # No need for parameter tuning because we have best parameters
            # Set elements of the pipeline, i.e. scaler and estimator
            scaler = StandardScaler()
            estimator = RandomForestRegressor(random_state=self.estimator_seeds[i], n_jobs=self.rforest_refit_jobs)

            # Create pipeline
            main_pipeline = Pipeline([
                ("scaler", scaler),
                ("estimator", estimator)
            ])    
            
            # No need for parameter tuning because we have best parameters
            # Fill best parameters to the to the estimator
            main_pipeline.set_params(**best_parameters_for_splits[self.split_seeds[i]][best_k])
            
            # Fit model on whole training data
            main_pipeline.fit(X_train_kfeats, y_train)
            
            # Evaluate the model and record the results
            pred = main_pipeline.predict(X_test_kfeats)
            # Record results in corresponding Experiment fields, mostly as named tuples
            # Classification performance
            ModelTestScores = collections.namedtuple("ModelTestScores", ["test_RMSE",
                                                            "test_explained_variance", "test_correlation"])
            model_test_scores = ModelTestScores(metrics.mean_squared_error(y_test, pred) ** 0.5,
                                               metrics.explained_variance_score(y_test, pred), 
                                                pearsonr(y_test, pred))
            # Record the best results
            test_results_for_splits[self.split_seeds[i]] = model_test_scores
            
        return (test_results_for_splits, dummy_for_splits, importances_for_splits, 
                best_parameters_for_splits, 
                tuning_seeds_for_splits, best_k)
    
#################################################################################################
# ModelingResults class
#################################################################################################
    
class ModelingResults(Modeling):
    
    def __init__(self, parent):
        if isinstance(parent, ModelingWithFeatureSelection):
            super().__init__(parent.name, parent.param_grid, parent.estimator_seeds, 
                                                  parent.split_seeds, parent.n_combinations, 
                                                  parent.kfolds, parent.rforest_jobs, 
                                                  parent.tuning_jobs, parent.scoring, parent.test_size)
            
            # RandomForest specific attributes
            self.n_combinations_importances = parent.n_combinations_importances
            self.kfolds_importances = parent.kfolds_importances
            
            # ElasticNet / StabilitySelection specific attributes
            self.ss_lambda_grid = parent.ss_lambda_grid
            self.ss_n_bootstrap_iterations = parent.ss_n_bootstrap_iterations
            self.ss_threshold = parent.ss_threshold
            self.ss_n_jobs = parent.ss_n_jobs
            
            # Create fields for storage of results
            # All fields are dictionaries with drugs as keys and dictonaries with split seeds
            # as keys as values
            self.performance_dict = {}
            self.dummy_performance_dict = {}
            self.importances_dict = {}
            self.best_params_dict = {}
            self.tuning_seeds_dict = {}
            self.best_k_dict = {}
        else:
            super().__init__(parent.name, parent.param_grid, parent.estimator_seeds, 
                                                  parent.split_seeds, parent.n_combinations, 
                                                  parent.kfolds, parent.rforest_jobs, 
                                                  parent.rforest_refit_jobs, parent.max_iter,
                                                  parent.tuning_jobs, parent.scoring, parent.test_size)
            
            # Create fields for storage of results
            # All fields are dictionaries with drugs as keys and dictonaries with split seeds
            # as keys as values
            self.performance_dict = {}
            self.dummy_performance_dict = {}
            self.best_params_dict = {}
            self.tuning_seeds_dict = {}
            self.data_shapes = {}
            self.cv_results_dict = {}
            
    def create_raw_results_df(self, drug_annotations_df=None, remove_duplicates=True):
        """Put results storage in dictonaries into one pandas DataFrame.
        
        Each row in the raw DataFrame will consist of drug, data split seed, and results. 
        The DataFrame stores just the raw results coming from dictonaries.
        
        Arguments:
            drug_annotations_df (DataFrame): list of drugs from GDSC. If parsed, it is used to
                        extract compounds target pathways.
        
        Returns:
            df (DataFrame): DataFrame with raw experiment results.
        """
        
        # Initialize future columns
        drug_names = []
        drug_ids = []
        split_seeds_columns = []
        test_rmses = []
        test_correlations = []
        corr_pvals = []
        dummy_rmses = []
        
        # If modeling was with feature selection, add a column with best feature number k
        if hasattr(self, "best_k_dict"):
            if len(self.best_k_dict) > 0:
                best_ks = []
                
        # If drug_annotations_df is parsed, extract target pathway information
        if drug_annotations_df is not None:
            target_pathways = []

        for drug_tuple in self.performance_dict:
            test_performance = self.performance_dict[drug_tuple]
            dummy_performance = self.dummy_performance_dict[drug_tuple]
            if hasattr(self, "best_k_dict"):
                if len(self.best_k_dict) > 0:
                    best_k = self.best_k_dict[drug_tuple]
                    # Check if modeling was with Stability Selection
                    if best_k <= 1.0:
                        corresponding_ks = []
                        for split_seed in self.best_params_dict[drug_tuple]:
                            for k, thresh in self.best_params_dict[drug_tuple][split_seed].keys():
                                if thresh == best_k:
                                    corresponding_ks.append(k)
                        best_k = int(np.mean(corresponding_ks))
            if drug_annotations_df is not None:
                pathway = drug_annotations_df[
                    drug_annotations_df["DRUG_ID"] == drug_tuple[1]]["TARGET_PATHWAY"].iloc[0]
                        
            for split_seed in test_performance.keys():
                test_results = test_performance[split_seed]
                dummy_results = dummy_performance[split_seed]

                # Fill the columns lists
                drug_names.append(drug_tuple[0])
                drug_ids.append(drug_tuple[1])
                split_seeds_columns.append(split_seed)
                test_rmses.append(test_results.test_RMSE)
                test_correlations.append(test_results.test_correlation[0])
                corr_pvals.append(test_results.test_correlation[1])
                dummy_rmses.append(dummy_results.test_RMSE)
                if hasattr(self, "best_k_dict"):
                    if len(self.best_k_dict) > 0:
                        best_ks.append(best_k)
                if drug_annotations_df is not None:
                    target_pathways.append(pathway)

        # Put column lists into DataFrame
        df = pd.DataFrame()
        df.insert(0, "Drug ID", drug_ids)
        df.insert(1, "Drug Name", drug_names)
        df.insert(2, "Split seed", split_seeds_columns)
        df.insert(3, "Model test RMSE", test_rmses)
        df.insert(4, "Model test correlation", test_correlations)
        df.insert(5, "Correlation pval", corr_pvals)
        df.insert(6, "Dummy test RMSE", dummy_rmses)
        df.insert(7, "Relative test RMSE", df["Dummy test RMSE"] / df["Model test RMSE"])
        if hasattr(self, "best_k_dict"):
            if len(self.best_k_dict) > 0:
                df.insert(8, "Best k", best_ks)
        if drug_annotations_df is not None:
            df.insert(8, "Target Pathway", target_pathways)
            
        if remove_duplicates:
            filepath = "/media/krzysztof/Nowy/Doktorat - Modelling drug efficacy in cancer/Projects/Results Assesment/Results and other files/drug_ids_to_keep_in_results.pkl"
            # Load list with drug ids to keep
            with open(filepath, "rb") as f:
                ids_to_keep = dill.load(f)
            return df[df["Drug ID"].isin(ids_to_keep)]
        return df
    
    def create_agg_results_df(self, columns_to_aggregate, metric_to_aggregate, drug_annotations_df=None,
                             remove_duplicates=True):
        """Compute DataFrame with results grouped by Drug ID and aggregated by mean."""
        # First, create the DF with raw results
        df = self.create_raw_results_df(drug_annotations_df, remove_duplicates)
        # Group by Drug ID
        df_grouped = df.groupby("Drug ID", as_index=False)
        # Aggregate by chosen metric
        agg_dict = {"Drug Name": lambda x: x.iloc[0]}
        for metric in columns_to_aggregate:
            agg_dict[metric] = metric_to_aggregate
        if drug_annotations_df is not None:
            agg_dict["Target Pathway"] = lambda x: x.iloc[0]
        return df_grouped.agg(agg_dict)
    
    def performance_barplots_single(self, metric, 
                                    columns_to_aggregate=["Model test RMSE", "Model test correlation", 
                                                          "Dummy test RMSE", "Relative test RMSE"],
                                    metric_to_aggregate="mean", drug_annotations_df=None,
                                    figsize=(35, 12), title_size=45, label_size=30, 
                                    tick_size=20, width=0.7, grid=True, half=1, 
                                    hline_width=2., save_directory=None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(metric + " for each drug", fontsize=title_size)
        ax.set_xlabel("Drug", fontsize=label_size)
        ax.set_ylabel(metric, fontsize=label_size)
        
        # Compute DataFrame with results
        df = self.create_agg_results_df(columns_to_aggregate, metric_to_aggregate, 
                                        drug_annotations_df)
    
        # Set up DataFrame slicing
        if half == 1:
            start_idx = 0
            end_idx = df.shape[0] // 2
        elif half == 2:
            start_idx = df.shape[0] // 2
            end_idx = df.shape[0]
        else:
            start_idx = 0
            end_idx = df.shape[0]

        df.sort_values(metric, ascending = False).iloc[start_idx:end_idx].plot(
            x = "Drug Name", y = metric ,kind="bar", ax = ax, width = width,
            figsize=figsize, fontsize=tick_size, grid = grid, legend=False)
        
        # If a metric is Relative Test RMSE, add a horziontal line at 1, 
        # represanting a baseline
        if metric == "Relative test RMSE":
            ax.axhline(y = 1.0, linewidth=hline_width, color="black")
            
        plt.tight_layout()
        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    def overall_boxplot_single(self, metric,aggregated=True, title="",
                       columns_to_aggregate=["Model test RMSE", "Model test correlation", 
                                                          "Dummy test RMSE", "Relative test RMSE"],
                                    metric_to_aggregate="mean", drug_annotations_df=None,
                                    figsize=(8, 6), title_size=25, label_size=20, 
                                    tick_size=18, grid=True, save_directory=None):
        
        
        # If aggregated is True, plot aggregated results
        if aggregated:
            # Compute DataFrame with aggregated results
            df = self.create_agg_results_df(columns_to_aggregate, metric_to_aggregate, 
                                            drug_annotations_df)
            # Actual plotting
            fig = plt.figure(figsize=figsize)
            plt.tick_params(labelsize=tick_size)
            plt.title(title, fontsize = title_size)
            if grid:
                plt.grid()

            sns.boxplot(x = df[metric], orient="v")
            plt.xlabel("", fontsize=label_size)
            plt.ylabel(metric, fontsize=label_size)
            
            plt.tight_layout()


            if save_directory:
                plt.savefig(save_directory)
            plt.show()
            
        # Otherwise, plot results for different data splits separately
        else:
            # Compute DataFrame with raw results
            df = self.create_raw_results_df(drug_annotations_df)
            
            # Actual plotting
            fig = plt.figure(figsize=figsize)
            plt.tick_params(labelsize=tick_size)
            plt.title(title, fontsize = title_size)
            if grid:
                plt.grid()

            sns.boxplot(x="Split seed", y=metric, data=df)
            plt.xlabel("Data split seed", fontsize=label_size)
            plt.ylabel(metric, fontsize=label_size)
            
            plt.tight_layout()

            if save_directory:
                plt.savefig(save_directory)
            plt.show()
        
    def extract_feature_ranks_per_split_ss(self, drug_tuple, split_seed, gene_expression_df):
        # Find appropriate selector
        selector = self.importances_dict[drug_tuple][split_seed]
        
        # Extract array containing max stabilty scores for features (should be a vector)
        scores_vector = selector.stability_scores_.max(axis=1)
        
        # Create a list with tuples (gene_id, max_score)
        feats_with_scores = [x for x in zip(gene_expression_df.ensembl_gene, scores_vector)]
        
        # Create dictionary of the form gene_id: rank
        ranks_dict = {}
        for rank, feat_tuple in enumerate(sorted(feats_with_scores, key=lambda x: x[1], reverse=True)):
            ranks_dict[feat_tuple[0]] = rank + 1
            
        # Create a final list of tuples (gene_id, rank) in the same order as in original gene_expression_df
        ranks_list = []
        for gene_id in gene_expression_df.ensembl_gene:
            ranks_list.append((gene_id, ranks_dict[gene_id]))
            
        return ranks_list
    
    def extract_feature_ranks_aggregated_over_splits_ss(self, drug_tuple, gene_expression_df):
        # Initialize array for storage of intermediate results
        ranks_vector = np.zeros(gene_expression_df.shape[0])
        # Iterate over data splits and add corresponding features ranks
        for split_seed in self.importances_dict[drug_tuple]:
            features_with_rank = self.extract_feature_ranks_per_split_ss(drug_tuple, split_seed,
                                                                                 gene_expression_df)
            ranks_vector = ranks_vector + np.array([x[1] for x in features_with_rank])
        # Divide ranks_vector by number of splits to get mean rank
        ranks_vector = ranks_vector / len(self.importances_dict[drug_tuple])
        
        # Add gene ids and return the results
        return [x for x in zip(gene_expression_df.ensembl_gene, ranks_vector)]
    
    def extract_topk_relevant_features_ss(self, drug_tuple, k, gene_expression_df, 
                                          just_gene_ids=True):
        # Compute genes along with corresponidng aggregated ranks over splits
        genes_with_ranks = self.extract_feature_ranks_aggregated_over_splits_ss(drug_tuple, 
                                                                    gene_expression_df)
        # If just_gene_ids is True, return only IDs corresponging to k
        # most relevant genes
        if just_gene_ids:
            return [x[0] for x in sorted(genes_with_ranks, key=lambda x: x[1])[:k]]
        # Otherwise, return gene IDs along with corresponding mean ranks
        else:
            return sorted(genes_with_ranks, key=lambda x: x[1])[:k]
    
    def extract_feature_ranks_per_split_rforest(self, drug_tuple, split_seed):
        # Extract proper vector with importance coeeficients
        feat_coeffs = self.importances_dict[drug_tuple][split_seed]
        
        # Create dictionary of the form gene_id: rank
        ranks_dict = {}
        for rank, feat_tuple in enumerate(sorted(feat_coeffs, key=lambda x: x[1], reverse=True)):
            ranks_dict[feat_tuple[0]] = rank + 1
            
        # Initialize final list with tuples (gene_id, rank), in the same order as in
        # original list
        ranks_list = []
        
        # Iterate over rank_dict and add entriess into rank_list
        for gene_id in [x[0] for x in feat_coeffs]:
            ranks_list.append((gene_id, ranks_dict[gene_id]))
            
        return ranks_list
    
    def extract_feature_ranks_aggregated_over_splits_rforest(self, drug_tuple):
        # Initialize array for storage of intermediate results
        ranks_vector = np.zeros(17737)
        # Iterate over data splits and add corresponding feature ranks
        for split_seed in self.importances_dict[drug_tuple]:
            features_with_rank = self.extract_feature_ranks_per_split_rforest(drug_tuple, 
                                                                              split_seed)
            ranks_vector = ranks_vector + np.array([x[1] for x in features_with_rank])
        
        # Divide ranks vector by number of splits to get mean ranks
        ranks_vector = ranks_vector / len(self.importances_dict[drug_tuple])
        
        # Add gene ids and return the results
        gene_ids = [x[0] for x in features_with_rank]
        return [x for x in zip(gene_ids, ranks_vector)]
    
    def extract_topk_relevant_features_rforest(self, drug_tuple, k, just_genes_ids=True):
        # Compute genes along with corresponding aggregated ranks over splits
        genes_with_ranks = self.extract_feature_ranks_aggregated_over_splits_rforest(drug_tuple)
        
        # If just_gene_ids is True, return only IDs corresponging to k
        # most relevant genes
        if just_genes_ids:
            return [x[0] for x in sorted(genes_with_ranks, key=lambda x: x[1])[:k]]
        # Otherwise, return gene IDs along with corresponding mean ranks
        else:
            return sorted(genes_with_ranks, key=lambda x: x[1])[:k]
        
    def reproduce_cv_results_vs_feat_numbers_ss(self):
        pass
    
    def reproduce_cv_results_vs_feat_numbers_rforest(self, drug_tuple, drug_annotations_df,
                                                    gene_expression_df, drug_response_df, 
                                                    rforest_jobs=1, tuning_jobs=1, log=False):
        # First, create corresponding DrugGenomeWide object
        # Get appropriate drug attribute from drug annotations
        row = drug_annotations_df[drug_annotations_df["DRUG_ID"] == drug_tuple[1]]
        
        gdsc_id = row["DRUG_ID"].iloc[0]
        name = row["DRUG_NAME"].iloc[0]
        targets = row["TARGET"].iloc[0].split(", ")
        target_pathway = row["TARGET_PATHWAY"].iloc[0]

        drug = DrugGenomeWide(gdsc_id, name, targets, target_pathway)
        print(drug)
        print(drug.targets)
        print(self.param_grid)
        
        # Extract full data
        data = drug.return_full_data(drug_response_df, gene_expression_df, data_combination=["expression"])
        
        if log:
            print("Reproducing results for drug {}, {}, data shape: {}.".format(
                drug.name, drug_gdsc_id, data.shape))

        # Extract features and labels
        y = data["AUC"]
        X = data.drop(["cell_line_id", "AUC"], axis=1)
        assert X.shape[1] == 17737 and X.shape[0] == y.shape[0]

        cv_results_over_splits = {}

        # Iterate over data splits
        c = 0
        for seed_of_interest in exp_results.performance_dict[drug_tuple]:
            if log:
                print()
                print("Reproducing for split seed:", seed_of_interest)

            # Split into train and test
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                               random_state=seed_of_interest)
            # Inportance vector
            importances = np.array([x[1] for x in exp_results.importances_dict[drug_tuple][seed_of_interest]])
            indices = importances.argsort()[::-1]

            # Initialize dictionary with cross-validation results over feature numbers k
            cv_scores = {}

            for k in exp_results.best_params_dict[drug_tuple][seed_of_interest]:
                # Extract reduced Xs
                X_train_reduced = self.selectKImportance(X_train, indices, k)
                X_test_reduced = self.selectKImportance(X_test, indices, k)

                # Setup the model
                scaler = StandardScaler()
                estimator = RandomForestRegressor(random_state=exp_results.estimator_seeds[c], 
                                                  n_jobs=rforest_jobs)

                pipe = Pipeline([
                    ("scaler", scaler),
                    ("estimator", estimator)
                ])

                # Set best params
                best_params = exp_results.best_params_dict[drug_tuple][seed_of_interest][k]

                pipe.set_params(**best_params)

                # Get cross-val score
                kfolds_results = model_selection.cross_val_score(pipe, X_train_reduced, y_train, 
                                                                 scoring = "neg_mean_squared_error", cv=self.kfolds,
                                                                n_jobs=tuning_jobs)
                res = (-np.mean(kfolds_results)) ** 0.5
                if log:
                    print("Done modeling with {} features".format(k))

                cv_scores[k] = res

            c += 1

            cv_results_over_splits[seed_of_interest] = cv_scores
            
        return cv_results_over_splits
        
        
    @staticmethod
    def comparative_df(experiments, flags, flag_name="Model",
                       columns_to_aggregate=["Model test RMSE", "Model test correlation", 
                                                          "Dummy test RMSE", "Relative test RMSE"],
                                    metric_to_aggregate="mean", drug_annotations_df=None,
                            remove_duplicates=True):
        # Compute aggregated DF with results for every experiment
        dfs = []
        for i in range(len(experiments)):
            df = experiments[i].create_agg_results_df(columns_to_aggregate, metric_to_aggregate, 
                                                      drug_annotations_df, remove_duplicates)
            dfs.append(df)
            
        # Find the intersection in drugs
        drugs_intersection = set(dfs[0]["Drug ID"])
        for i in range(1, len(dfs)):
            df = dfs[i]
            drugs_intersection = drugs_intersection.intersection(df["Drug ID"])
        drugs_intersection = list(drugs_intersection)
        
        # Create a list of DataFrames with common drugs
        dfs_common = [1 for x in range(len(experiments))]
        for i in range(len(dfs)):
            df_common = dfs[i][dfs[i]["Drug ID"].isin(drugs_intersection)]
            assert df_common.shape[0] == len(drugs_intersection)
            # Add a column with appropriate flags
            flags_column = [flags[i]] * df_common.shape[0]
            df_common[flag_name] = flags_column
            dfs_common[i] = df_common
        
        # Concatenate all DataFrames
        comparative_df = pd.concat(dfs_common, axis=0)
        # Sort final DF bu drug IDs
        return comparative_df.sort_values("Drug ID")
    
        
    @staticmethod
    def barplot_from_comparative_df(data, x, y, hue=None, order=None, hue_order=None, title="",
                                    figsize=(35, 12), title_size=45, label_size=30, 
                                    tick_size=20, width=0.7, grid=True, half=1, 
                                    hline_width=2., legend_fontsize=25, save_directory=None,
                                     xticks_off=False, ax=None):
        
        params = {"legend.fontsize": legend_fontsize,
                 "legend.handlelength": 2}
        plt.rcParams.update(params)
        
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(x, fontsize=label_size)
        ax.set_ylabel(y, fontsize=label_size)
        
        sns.barplot(x, y, hue=hue,
                       data=data,
                    order=order,
                   hue_order=hue_order,
                   ax=ax)
        
        plt.xticks(rotation='vertical')
        plt.tick_params("both", labelsize=tick_size)
        if xticks_off:
            ax.set_xlabel("")
            plt.xticks([], [])
            
        plt.legend(title = "")
        
        # If a metric is Relative Test RMSE, add a horziontal line at 1, 
        # represanting a baseline
        if y == "Relative test RMSE":
            ax.axhline(y = 1.0, linewidth=hline_width, color="black")
            
        plt.tight_layout()

        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
    def catplot_from_comparative_df(data, x, y, hue=None, order=None, hue_order=None, col=None, row=None,
                                kind="bar", title="",
                                    height=5, aspect=2., title_size=45, label_size=35, 
                                    tick_size=25, width=0.7, grid=False, half=1, 
                                    hline_width=2., legend_fontsize=25, save_directory=None,
                                     xticks_off=False, ax=None, legend=True, legend_out=False, marker_size=None):
        
        params = {"legend.fontsize": legend_fontsize,
                "legend.handlelength": 2}
        plt.rcParams.update(params)
        
        arguments = {"x": x, "y": y, "hue": hue, "data": data, "kind": kind,
                    "order":order, "hue_order": hue_order, "col":col, "row": row,
                    "height": height, "aspect": aspect, "ax": ax,
                    "legend_out": legend_out, "legend": legend}
        
        if kind in ("bar", "violin", "box"):
            g = sns.catplot(**arguments)
        else:
            g = sns.catplot(**arguments, s=marker_size)

        plt.tick_params("both", labelsize=tick_size)
        plt.xticks(rotation="vertical")

        plt.xlabel(x, fontsize=label_size)
        plt.ylabel(y, fontsize=label_size)
        plt.title(title, fontsize=40)

        if xticks_off:
            ax.set_xlabel("")
            plt.xticks([], [])
        if grid:
            plt.grid()
        plt.legend(title = "")

        # If a metric is Relative Test RMSE, add a horziontal line at 1, 
        # represanting a baseline
        if y == "Relative test RMSE":
            plt.axhline(y=1., color='b', linewidth=hline_width)

        plt.tight_layout()


        if save_directory:
            plt.savefig(save_directory)
            
        plt.close(g.fig)
        plt.show()
        
    @staticmethod
    def sort_drugs_by_comparable_performance(comparative_df, model_to_compare, metric, flag_name="Model"):
        results = []
        for drug_name in comparative_df["Drug Name"].unique():
            current_df = comparative_df[comparative_df["Drug Name"] == drug_name]
            baseline_performance = current_df[current_df[flag_name] == model_to_compare][
                metric
            ].iloc[0]
            # Find the best model among rest
            if metric == "Model test RMSE":
                best_metric = 1.
                for model in current_df[flag_name].unique():
                    if model != model_to_compare:
                        performance = current_df[current_df[flag_name] == model][metric].iloc[0]
                        if performance < best_metric:
                            best_metric = performance
            else:
                best_metric = 0.0
                for model in current_df[flag_name].unique():
                    if model != model_to_compare:
                        performance = current_df[current_df[flag_name] == model][metric].iloc[0]
                        if performance > best_metric:
                            best_metric = performance

            # Establish relative metric
            relative_performance = baseline_performance / best_metric

            # Add to results
            results.append((drug_name, relative_performance))

        # Sort the results
        if metric == "Model test RMSE":
            order = sorted(results, key=lambda x: x[1])
        else:
            order = sorted(results, key=lambda x: x[1], reverse=True)
            
        return order

        
    @staticmethod
    def boxplot_from_comparative_df(data, x, y, hue=None, order=None, title="",
                                   figsize=(8, 6), title_size=25, label_size=20, 
                                    tick_size=18, grid=True, rotation=-90, save_directory=None,
                                    xticks_off=False):
        # Actual plotting
        fig = plt.figure(figsize=figsize)
        plt.tick_params(labelsize=tick_size)
        plt.title(title, fontsize = title_size)
        plt.grid()

        sns.boxplot(x, y, data=data, hue=hue, order=order)
        plt.xlabel("", fontsize=label_size)
        plt.ylabel(y, fontsize=label_size)
        plt.xticks(rotation=rotation)
        
        if xticks_off:
            plt.xticks([], [])
        
        plt.tight_layout()

        
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
        
        
    @staticmethod
    def filterout_good_drugs(data, rel_rmse_threshold=1.01, correlation_threshold=0.0, flag_name="Model"):
        # Extract drug IDs for which one of the models exceeded threshold
        good_drug_ids = []
        for drug_id in data["Drug ID"].unique():
            current_df = data[data["Drug ID"] == drug_id]
            for model in current_df[flag_name]:
                current_rel_rmse = current_df[current_df[flag_name] == model][
                    "Relative test RMSE"].iloc[0]
                current_correlation = current_df[current_df[flag_name] == model][
                    "Model test correlation"].iloc[0]
                if (current_rel_rmse > rel_rmse_threshold) and (current_correlation > correlation_threshold):
                    good_drug_ids.append(drug_id)
                    break
        
        # Return DataFrame containing only well performing drugs
        return data[data["Drug ID"].isin(good_drug_ids)]

    @staticmethod
    def get_best_results(overall_agg_dataframe, metric="Model test correlation"):
        ides = []
        names = []
        corrs = []
        rel_rmses = []
        test_rmses = []
        dummy_rmses = []
        corr_pvals = []
        experiment_names = []
        pathways = []
        # Iterate over drugs
        for drug_id in overall_agg_dataframe["Drug ID"].unique():
            df = overall_agg_dataframe[overall_agg_dataframe["Drug ID"] == drug_id]
            # Find row corresponding the best metric
            if metric == "Model test correlation" or metric == "Relative test RMSE":
                max_idx = df[metric].idxmax()
                max_row = df.loc[max_idx]
            else:
                max_idx = df[metric].idxmin()
                max_row = df.loc[max_idx]

            name = max_row["Drug Name"]
            corr = max_row["Model test correlation"]
            rel_rmse = max_row["Relative test RMSE"]
            test_rmse = max_row["Model test RMSE"]
            experiment = max_row["Model"]
            pathway = max_row["Target Pathway"]
            corr_pval = max_row["Correlation pval"]
            dummy_rmse = max_row["Dummy test RMSE"]

            ides.append(drug_id)
            names.append(name)
            corrs.append(corr)
            rel_rmses.append(rel_rmse)
            test_rmses.append(test_rmse)
            experiment_names.append(experiment)
            pathways.append(pathway)
            corr_pvals.append(corr_pval)
            dummy_rmses.append(dummy_rmse)

        # Put into DataFrame
        df_best_results = pd.DataFrame()

        df_best_results["Drug ID"] = ides
        df_best_results["Drug Name"] = names
        df_best_results["Model test correlation"] = corrs
        df_best_results["Correlation pval"] = corr_pvals
        df_best_results["Relative test RMSE"] = rel_rmses
        df_best_results["Model test RMSE"] = test_rmses
        df_best_results["Dummy test RMSE"] = dummy_rmses
        df_best_results["Corresponding experiment"] = experiment_names
        df_best_results["Target Pathway"] = pathways

        return df_best_results.sort_values("Model test correlation", ascending=False).reset_index(drop=True)
    
    @staticmethod
    def filter_bad_drugs(best_results_df, overall_raw_results_df, rel_rmse_threshold=1.0):
        """Return list of drug IDs with sufficient performance."""
        # Initialize list of drugs with corresponding poor result
        bad_drugs = []
        # Iterate over drugs
        for entry in best_results_df.iterrows():
            # Extract corresponding drug ID and best setting
            drug_id = entry[1]["Drug ID"]
            best_setting = entry[1]["Corresponding experiment"]

            # Extract not aggregated data corresponding to the drug and setting
            df = overall_raw_results_df[
                (overall_raw_results_df["Drug ID"] == drug_id) & (overall_raw_results_df["Model"] == best_setting)
            ]
    #         if df["Relative test RMSE"].median() < rel_rmse_threshold:
    #             bad_drugs.append(drug_id)
            # Iterate over rows in extracted data
            for row in df.iterrows():
                # Extract relative RMSE
                rel_rmse = row[1]["Relative test RMSE"]
                # If relative RMSE is below treshold for even one of the seeds, exclude this drug
                if rel_rmse < rel_rmse_threshold:
                    bad_drugs.append(drug_id)
                    break
        # Return all drug IDs excluding bad drugs
        return bad_drugs
        # return set(best_results_df["Drug ID"].unique()).difference(set(bad_drugs))
        
    
    @staticmethod
    def extract_top_k_drugs_overall(best_score_dataframe, k, metric="Model test correlation"):
        # Sort DataFrame with best score
        df_sorted = best_score_dataframe.sort_values(metric, ascending=False)
        return df_sorted.iloc[:k]
    
    @staticmethod
    def prepare_data_for_scatter_plots(full_data_gw, full_data_restricted, data_split):
        # Establish intersection in terms of cell lines
        cl_intersection = set(full_data_gw.cell_line_id).intersection(set(full_data_restricted.cell_line_id))
        cl_intersection = list(cl_intersection)

        # Extract common data for both dataframes
        data_gw = full_data_gw[full_data_gw["cell_line_id"].isin(cl_intersection)]
        data_gw = data_gw.sort_values("cell_line_id")

        data_restricted = full_data_restricted[full_data_restricted["cell_line_id"].isin(cl_intersection)]
        data_restricted = data_restricted.sort_values("cell_line_id")

        # Extract features and labels
        y_both = data_gw["AUC"]
        X_gw = data_gw.drop(["cell_line_id", "AUC"], axis=1)

        X_restricted = data_restricted.drop(["cell_line_id", "AUC"], axis=1)

        # Split into train and test
        X_train_gw, X_test_gw, y_train_both, y_test_both = model_selection.train_test_split(X_gw, y_both, test_size=0.3,
                                                                                       random_state=data_split)
        X_train_restricted, X_test_restricted, _, _ = model_selection.train_test_split(
                X_restricted, y_both, test_size=0.3,
                random_state=data_split)
        return X_train_gw, X_test_gw, X_train_restricted, X_test_restricted, y_train_both, y_test_both
    
    @staticmethod
    def model_for_scatter_plots(X_train, X_test, y_train, y_test, estimator, param_grid,
                                tuning_seed=0, n_combinations=30,
                                scoring="neg_mean_squared_error", 
                                n_jobs=1, kfolds=3,
                                verbose_cv=1, log=False):
        # Setup modeling pipeline
        scaler = StandardScaler()

        # Create a pipeline
        pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])

        # Setup best grid search
        grid = model_selection.RandomizedSearchCV(pipeline, param_grid, 
                                                  n_iter=n_combinations,
                                                  scoring=scoring,
                                                  n_jobs=n_jobs,
                                                  cv=kfolds,
                                                  random_state=tuning_seed,
                                                  verbose=verbose_cv)
                                                  
        
        # Fit the grid on training data
        grid.fit(X_train, y_train)

        # Predict on test set
        preds = grid.predict(X_test)

        if log:
        # Evaluate
            test_corr_gw = pearsonr(y_test, preds)
            test_rmse_gw = metrics.mean_squared_error(y_test, preds) ** 0.5

            print("Test correlation:", test_corr_gw)
            print("Test RMSE:", test_rmse_gw)
            print(grid.best_params_)
        return preds, grid
    
    @staticmethod
    def model_for_scatter_plots_with_best_params(X_train, X_test, y_train, y_test, estimator, best_params,
                           log=False):
        
        # Setup modeling pipeline
        scaler = StandardScaler()

        # Create a pipeline
        pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])

        # Setup best parameters
        pipeline.set_params(**best_params)
        
        # Fit the grid on training data
        pipeline.fit(X_train, y_train)

        # Predict on test set
        preds = pipeline.predict(X_test)

        if log:
        # Evaluate
            test_corr = pearsonr(y_test, preds)
            test_rmse = metrics.mean_squared_error(y_test, preds) ** 0.5

            print("Test correlation:", test_corr)
            print("Test RMSE:", test_rmse)
        return preds
    
    
    # Functions for extracting feature importances
    @staticmethod
    def extract_feature_importances_per_split(X, y, estimator, split_seed, best_params):
        # Split into training and test set
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                           random_state=split_seed)
        # Build a pipeline
        scaler = StandardScaler()

        pipeline = Pipeline([
            ("scaler", scaler),
            ("estimator", estimator)
        ])
        # Setup best parameters
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        # Distinguish between RF and EN
        if hasattr(estimator, "n_estimators"):   # RForest case
            importances = pipeline.named_steps["estimator"].feature_importances_
            return importances
        else:   # ElasticNet case
            coeffs = pipeline.named_steps["estimator"].coef_
            intercept = pipeline.named_steps["estimator"].intercept_
            return coeffs
        
    def extract_feature_importances_across_splits(self, full_data, drug_tuple, algorithm, log=False):
        # Extract features and labels
        y = full_data["AUC"]
        X = full_data.drop(["cell_line_id", "AUC"], axis=1)
        assert X.shape[0] == y.shape[0]

        # Initialize holder for coefficients
        importances_sum = np.zeros(len(X.columns))

        # Iterate over split seeds
        for split_seed in self.split_seeds:
            # Extract corresponding estimator seed
            estimator_seed = self.estimator_seeds[self.split_seeds.index(split_seed)]
            estimator = clone(algorithm)
            estimator.random_state = estimator_seed
            # Extract corresponding best parameters
            best_params = self.best_params_dict[drug_tuple][split_seed]
            
            # Extract feature importances
            importances = self.extract_feature_importances_per_split(X, y, estimator, split_seed, best_params)
            importances_sum = importances_sum + importances
            if log:
                print(split_seed)
                print(importances[:5])
                print()
        return importances_sum / len(self.split_seeds), X.columns
    
    @staticmethod
    def sort_feature_importances(importances, cols, map_ensembl_to_hgnc=None,
                                normalize=False):
        """Take a vector with importance coeeficents and feature names and return
        a list of tuples (name, importance) sorted descending by importances"""
        if map_ensembl_to_hgnc:
            mapped_cols = []
            for col in cols:
                if col in map_ensembl_to_hgnc:
                    mapped_cols.append(map_ensembl_to_hgnc[col])
                else:
                    mapped_cols.append(col)
            if normalize:
                imp_sum = sum(map(abs, importances))
                importances = [x / imp_sum for x in importances]
            res = [x for x in zip(mapped_cols, importances)]
            return sorted(res, key=lambda x: abs(x[1]), reverse=True)
        
        if normalize:
            imp_sum = sum(map(abs, importances))
            importances = [x / imp_sum for x in importances]
        res = [x for x in zip(cols, importances)]
        return sorted(res, key=lambda x: abs(x[1]), reverse=True)
    
    @staticmethod
    def check_feature_types(feats):
        types = set()
        for feat in feats:
            if feat[-3:] == "mut":
                types.add("mutation")
            elif feat[-3:] == "exp" or feat[:4] == "ENSG":
                types.add("expression")
            elif feat[:6] == "Tissue":
                types.add("tissue")
            elif feat[:3] == "cna":
                types.add("CNV")
            else:
                types.add("signatures")

            if len(types) == 5:
                break
        return types
    
    @staticmethod
    def count_feat_types(feats, feat_name):
        s = 0
        if feat_name == "expression":
            for feat in feats:
                if feat[-3:] == "exp" or feat[:4] == "ENSG":
                    s += 1
        elif feat_name == "mutation":
            for feat in feats:
                if feat[-3:] == "mut":
                    s += 1
        elif feat_name == "tissue":
            for feat in feats:
                if feat[:6] == "Tissue":
                    s += 1
        elif feat_name == "CNV":
            for feat in feats:
                if feat[:3] == "cna":
                    s += 1
        else:
            for feat in feats:
                if feat[-3:] == "mut":
                    pass
                elif feat[-3:] == "exp" or feat[:4] == "ENSG":
                    pass
                elif feat[:6] == "Tissue":
                    pass
                elif feat[:3] == "cna":
                    pass
                else:
                    s += 1
        return s
                
class ModelingWithEqualRights(object):
    def __init__(self, param_grid, split_seed):
        self.param_grid = param_grid
        self.split_seed = split_seed
        
        # Dictionaries with data
        self.estimator_seeds = {}
        self.tuning_seeds = {}
        self.exp_alone_results = {}
        self.all_data_results = {}
        self.grids = {}
        
    def create_results_df(self):
        drug_ides = []
        drug_names = []
        test_rmses = []
        test_corrs = []
        test_corrs_pvals = []
        train_rmses = []
        train_corrs = []
        models = []
        
        for name, ide in self.exp_alone_results:
            dic = self.exp_alone_results[(name, ide)]
            
            drug_ides.append(ide)
            drug_names.append(name)
            
            test_rmse = dic["Test RMSE"]
            test_corr = dic["Test corr"][0]
            test_corr_pval = dic["Test corr"][1]
            train_rmse = dic["Train RMSE"]
            train_corr = dic["Train corr"][0]
            
            test_rmses.append(test_rmse)
            test_corrs.append(test_corr)
            test_corrs_pvals.append(test_corr_pval)
            train_rmses.append(train_rmse)
            train_corrs.append(train_corr)
            
            models.append("Expression alone")
            
        for name, ide in self.all_data_results:
            dic = self.all_data_results[(name, ide)]
            
            drug_ides.append(ide)
            drug_names.append(name)
            
            test_rmse = dic["Test RMSE"]
            test_corr = dic["Test corr"][0]
            test_corr_pval = dic["Test corr"][1]
            train_rmse = dic["Train RMSE"]
            train_corr = dic["Train corr"][0]
            
            test_rmses.append(test_rmse)
            test_corrs.append(test_corr)
            test_corrs_pvals.append(test_corr_pval)
            train_rmses.append(train_rmse)
            train_corrs.append(train_corr)
            
            models.append("All data")
        
        df = pd.DataFrame()
        df["Drug ID"] = drug_ides
        df["Drug Name"] = drug_names
        df["Test RMSE"] = test_rmses 
        df["Test correlation"] = test_corrs
        df["Correlation pval"] = test_corrs_pvals
        df["Train RMSE"] = train_rmses
        df["Train correlation"] = train_corrs
        df["Model"] = models
        
        return df.sort_values("Drug ID").reset_index(drop=True)

#################################################################################################
# DrugOldCCLEandCTRP class (old data from CCLE)
#################################################################################################                            
                            
class DrugOldCCLEandCTRP(object):
    """Basic class representing drug with data from CTRP v2 dataset"""
    ctrp_exp_id_to_cl_name_mapper = None
    def __init__(self, name, ctrp_master_cpd_id, targets=None):
        self.name = name
        self.ctrp_master_cpd_id = ctrp_master_cpd_id
        if targets:
            self.targets = targets
    
    def extract_drug_response_data(self, drug_response_df):
        """Extract drug sensitivity data concerning this drug."""
        # DataFrame with data just for this drug
        df = drug_response_df[drug_response_df.master_cpd_id == self.ctrp_master_cpd_id]
        # Extract columns of interest
        df = df[["experiment_id", "area_under_curve"]]
        # Map experiment ID to cell line name
        if self.ctrp_exp_id_to_cl_name_mapper:
            df.insert(1, "cell_line_name", df["experiment_id"].map(self.ctrp_exp_id_to_cl_name_mapper))
            df = df.dropna(subset=["cell_line_name"])
        else:
            raise ValueError("Class variable ctrp_exp_id_to_cl_name_mapper is not filled")
        return df.reset_index(drop=True)
    
    def extract_full_data(self, drug_response_df, gene_expression_df=None, mutation_df=None,
                         tissue_df=None):
        """Extract design matrix for that drug, i.e. it's response data (target variable) from
        CTRP v2 along with genomic features for cell lines which were screened with this drug
        (from CCLE)"""
        # First, compute drug response data
        response_df = self.extract_drug_response_data(drug_response_df)
        # Establish genomic features
        genomic_dfs = []
        if type(gene_expression_df) == pd.core.frame.DataFrame:
            expression_df = self.extract_mRNA_expression_data_all_cell_lines(gene_expression_df)
            # Add "exp" suffix to all features
            expression_df.columns = [x + "_exp" for x in expression_df.columns]
            # Add column with cell line names to expression data
            expression_df.insert(0, "cell_line_name", expression_df.index)
            expression_df = expression_df.reset_index(drop=True)
            # Add this DF to list of genomic data
            genomic_dfs.append(expression_df)
        if type(mutation_df) == pd.core.frame.DataFrame:
            mutation_df = self.extract_mutation_data_all_cell_lines(mutation_df)
            # Add "mut" suffix to all features
            mutation_df.columns = [x + "_mut" for x in mutation_df.columns]
            # Add column with cell line names to mutation data
            mutation_df.insert(0, "cell_line_name", mutation_df.index)
            mutation_df = mutation_df.reset_index(drop=True)
            # Add this DF to list of genomic data
            genomic_dfs.append(mutation_df)
        if type(tissue_df) == pd.core.frame.DataFrame:
            tissue_df = self.extract_tissue_data_all_cell_lines(tissue_df)
            # Add column with cell line names to tissue data
            tissue_df.insert(0, "cell_line_name", tissue_df.index)
            tissue_df = tissue_df.reset_index(drop=True)
            # Add this DF to list of genomic data
            genomic_dfs.append(tissue_df)
        # Establish common cell lines across all datatypes
        cl_intersection = set(response_df.cell_line_name)
        for genomic_df in genomic_dfs:
            cl_intersection = cl_intersection.intersection(set(genomic_df.cell_line_name))
        # Extract data only for common cell lines
        response_df = response_df[response_df.cell_line_name.isin(cl_intersection)]
        response_df = response_df.sort_values("cell_line_name").reset_index(drop=True)
        stripped_dfs = []
        for genomic_df in genomic_dfs:
            df = genomic_df[genomic_df["cell_line_name"].isin(cl_intersection)]
            df = df.sort_values("cell_line_name")
            df = df.drop("cell_line_name", axis=1).reset_index(drop=True)
            stripped_dfs.append(df)
        full_data = pd.concat([response_df] + stripped_dfs, axis=1)
        return full_data
            
    @staticmethod    
    def extract_mRNA_expression_data_all_cell_lines(gene_expression_df):
        """Extract mRNA data from CCLE for all available cell lines."""
        # Transpose raw expression DF
        df = gene_expression_df.transpose()
        # Set gene names as columns
        df.columns = df.loc["#"]
        # Drop unnecessary rows and columns
        df = df.drop(["#", "#.1", "CellLine"])
        df = df.drop(["#", "GeneSym"], axis=1)
        # Current DF has gene symbols as columns and cell line names as row index
        # Convert all values to floats
        df = df.apply(pd.to_numeric)
        return df
    
    @staticmethod    
    def extract_mutation_data_all_cell_lines(mutation_df):
        """Extract all available mutation binary calls from CCLE for all cell lines"""
        df = mutation_df.transpose()
        # Set gene names as columns
        df.columns = df.loc["#"]
        # Drop unnecessary rows and columns
        df = df.drop(["#", "#.1", "CellLine"])
        df = df.drop(["#", "GeneSym"], axis=1)
        # Current DF has gene symbols as columns and cell line names as row index
        # Convert all values to integers
        df = df.apply(pd.to_numeric)
        df = df.astype(int)
        return df
    
    @staticmethod
    def extract_tissue_data_all_cell_lines(gene_expression_df):
        """Get tissue type data from CCLE. Data is extracted from the same
        DataFrame as gene expression"""
        # Get just tissue data from expression DataFrame
        df = pd.DataFrame(gene_expression_df.loc[0])
        # Drop unnecessary rows and rename th only column
        df = df.drop(["#", "#.1", "CellLine"])
        df.columns = ["Tissue"]
        # Dummy encode tissue types
        df = pd.get_dummies(df)
        return df
                            
    