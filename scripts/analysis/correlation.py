#!/usr/bin/env python3
# Author: Sara Tolosa Alarcón

import pandas as pd
import os
from scipy import stats                 
from itertools import combinations
from basic import amino_acids, mutant_dictionary, index_aa_dictionary
from distances import Distances


class CorrelationAnalysis:

    def load_experimental_data(self, exp_data_path=None):
        """ Load the experimental curated data. It should be a csv file with the following columns:
                - Pos: Position of sequence
                - WT_AA: Wild type aminoacid
                - Mut: Mutant aminoacid
                - fitness: experimental data (e.g. abundance, binding, etc.)
        """

        df_exp = pd.read_csv(exp_data_path)
        if df_exp["Pos"][0] == 0:
            df_exp["Pos"] +=1

        return df_exp
    
    def extract_diff(self, tensor, aa_dictionary, token, mutation, is_float=False):
        """ For a given 2D tensor, this function helps you to extract a value (or a tensor)
        of a given token and mutation according to a desired dictionary which stores the 
        position of mutant amino acids.
        
        --------------------------------------------
        Parameters:
                - tensor (numpy.ndarray): The 2D tensor from which to extract the value.
                - aa_dictionary (dict): A dictionary storing the position of mutant amino acids.
                - token (int): The index of the token for extraction.
                - mutation (str): The mutant amino acid for which the value is extracted.
                - float (bool, optional): If True, converts the extracted value to a float. Default is False.
            
            Returns:
                - The extracted value from the tensor.
            --------------------------------------------
        """
        mutation = mutation.upper()
        index_mutation = aa_dictionary[mutation]
        value = tensor[token, index_mutation].numpy()
        if is_float:
            value = float(value)

        return value
    
    
    def tensor_to_df_model(self, data, mutated_sequence, aa_list=amino_acids):
        """ Transform tensor data into a dataframe.
        --------------------------------------------
        Parameters:
                - dictionary_of_tensors: dictionary with the tensors to be transformed into a dataframe.
                - mutated_sequence: sequence of the protein with the mutations.
                - aa_list: list of amino acids to be used.
        --------------------------------------------
        """
        
        # 1. First, create an empty dataframe based on the mutated sequence and amino acid list used.
        mutant_dict = mutant_dictionary(mutated_sequence, aa_list)
        df_model = pd.DataFrame.from_dict(mutant_dict.items())

        # Transform the second column, which contains list of Mutant aminoacid for each WT_aa,
        # and create a row for each element of that list, duplicating the values of the first column (WT_AA)
        df_model = df_model.explode(1)
        # add a Position column with the index of the previous dataframe
        df_model["Pos"] = df_model.index + 1

        # Rename the columns
        df_model.rename(columns={0:"WT", 1:"Mut"}, inplace=True)


        # 2. Extract the values of the tensors for each position and mutation
        for tensor_name, value in data.items():
            valuelist=[]
            for i in range(len(mutated_sequence)):
                for pos,aa in enumerate(aa_list):

                    diff_extraction= self.extract_diff(tensor=value, 
                                                       aa_dictionary=index_aa_dictionary(aa_list),
                                                       token=i, mutation=aa,
                                                       is_float=True)
                    valuelist.append(diff_extraction)

            df_model[tensor_name] = valuelist
              
        return df_model
    

    def map_experimental_model_data(self, df_exp, df_model):
        """ Map both experimental and model dataframes according to Pos column.
        --------------------------------------------
        Parameters:
                - df_exp: experimental dataframe
                - df_model: model dataframe
        --------------------------------------------
        """
        
        df_result = pd.merge(df_model, df_exp, how="inner", on= ["Pos","Mut"], suffixes=('_model', '_exp'))
        
        return df_result
    
    
    def calculate_correlations(self, df, list_variables=list, fixed_variable=None):
        """
        Calculate the correlation between data pairs from a DataFrame and returns
        the correlation coefficient (R) and the p-value for each pair.
        --------------------------------------------
        Parameters:
                - df (pandas.DataFrame): DataFrame from which we want to extract the data and perform the correlation analysis.
                - list_variables (numpy.ndarray): List of variables to compare.
                - fixed_variable (str, optional): If provided, fix this variable and only do combinations with it.

        Returns:
                - A dataframe with all the correlation coefficients and p-values of each pair of variables.
        --------------------------------------------     
        """

        # 1. Transform the dataframe values into a dictionary of arrays:
        # Key: variable name
        # Value: array of values for that variable
        array_dictionary = {}
        
        for var in list_variables:
            array_dictionary[f"{var}"] = df[var].to_numpy()

        # 2. Correlation analysis:
        corr_rows=[]
        # Iterate over all the combinations of variables:
        for var1, var2 in combinations(array_dictionary.keys(), 2):

            if fixed_variable is not None and fixed_variable != var1 and fixed_variable != var2:
                continue  # In order to skip combinations that don't involve the fixed variable

            array1, array2 = array_dictionary[var1], array_dictionary[var2]

            # Perform Pearson and Spearman correlation tests:
            R_p, pvalue_p = stats.pearsonr(array1, array2, alternative="two-sided")
            R_sp, pvalue_sp = stats.spearmanr(array1, array2, alternative="two-sided")

            # For each pair of variables, append the results to the list corr_rows:
            corr_rows.append({"array_1": var1, "array_2": var2, "R_pearson": R_p, "pvalue_pearson": pvalue_p,
                                "R_spearman": R_sp, "pvalue_spearman": pvalue_sp})

        # 3. Transform the list into a dataframe:
        df_output_correlations = pd.DataFrame(corr_rows)

        return df_output_correlations
      

    def run_correlation_analysis(self, 
                                     exp_data_path, 
                                     tensor, 
                                     analyses, 
                                     distance_list, 
                                     mutated_seq, 
                                     outpath, 
                                     file_name,
                                     interface,
                                     interface_residue_list):
        """ Perform the correlation analysis between the experimental and model data.
        """

        # 1. Load experimental data
        exp_data = self.load_experimental_data(exp_data_path)
        
        # 2. Load our tensor data as a Distances class
        dis = Distances(tensor, 
                        mutated_ssequence=mutated_seq, 
                        aa_list=amino_acids, 
                        interface=interface, 
                        interface_residue_list=interface_residue_list)

        # 3. Perform global or positional analysis
        results_dictionary={}

        for d in distance_list:
            for a in analyses:

                if a =="global":
                    results_dictionary[f"{a}_analysis_{d}"] = dis.global_analysis(distance=d)

                elif a =="positional":
                    results_dictionary[f"{a}_analysis_{d}"] = dis.positional_analysis(distance=d)
        
        # 4. Define the directory path to store all the results
        # and the file names
        df_dir = os.path.join(outpath, "results_corr")
        if not os.path.isdir(df_dir):
            os.makedirs(df_dir)
        
        f_model = f"{df_dir}/df_{file_name}.csv"
        f_merged = f"{df_dir}/df_model_exp_{file_name}.csv"
        f_corr = f"{df_dir}/df_correlation_analysis_{file_name}.csv"
        f_mean = f"{df_dir}/df_meanPos_corr_analysis_{file_name}.csv"
                          

        # 5. Transform the tensors into a dataframe  
        df_model = self.tensor_to_df_model(data=results_dictionary, 
                                           mutated_sequence=mutated_seq, 
                                           aa_list=amino_acids)
        
        df_model.to_csv(f_model, index=False)

        # 6. Map experimental and model data and download the csv
        df_model_experimental_data = self.map_experimental_model_data(df_exp=exp_data, 
                                                                      df_model=df_model)
        df_model_experimental_data.to_csv(f_merged, index=False)


        # 7. Calculate the correlations between model and experimental data
        columns = df_model_experimental_data.select_dtypes(include=["float64","int64"]).columns # Define the columns used to make the analysis
        columns = columns.drop("Pos") # Remove the column Pos from columns list
        
        # 8. Perform general correlation analysis
        df_correlation = self.calculate_correlations(df_model_experimental_data, columns, fixed_variable="fitness")                                                      
        df_correlation.to_csv(f_corr, index=False)  

        # 9. Perform the correlation analysis using the mean position        
        df_mean_Pos = df_model_experimental_data.groupby("Pos")[columns].mean().reset_index()
        df_mean_Pos_analysis = self.calculate_correlations(df_mean_Pos, columns, fixed_variable="fitness")
        df_mean_Pos_analysis.to_csv(f_mean, index=False)
        
        return df_model_experimental_data, df_correlation, df_mean_Pos_analysis