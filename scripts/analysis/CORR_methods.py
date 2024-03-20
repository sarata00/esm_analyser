#!/usr/bin/env python3

import pandas as pd                     
from scipy import stats                 
from itertools import combinations
import basic as bs
import matplotlib.pyplot as plt

amino_acids = bs.amino_acids

def extract_diff(tensor, aa_dictionary, token, mutation, is_float=False):
   """
   For a given 2D tensor, this function helps you to extract a value (or a tensor)
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

    Example:
        - You have the following matrix, where rows are the mutant amino acids and columns the different
        tokens of your protein sequence (aka your protein residues)

            |    1    2    3 ... 23 ... N  |
            | A  0   0.5  0.1   0.75    X  |
            | C  0   0.2   0      1     X' |
            | D  0.1  0    1     1.1    X''|


    > extract_diff(tensor, aa_dictionary, 23, A, is_float=True)
    >> output = 0.75

    By this you will extract the value of the coordinate (23, A) where 23 is the token of the protein sequence
    and "A" is the amino acid by which that positon has been mutated.
   """

   mutation = mutation.upper()
   
   index_mutation = aa_dictionary[mutation]

   value = tensor[token, index_mutation].numpy()

   if is_float:
      value = float(value)

   return value


def correlation_analysis(array_1, array_2, test="pearson", alternative="two-sided", verbose=False):
   """
   Calculate the correlation between two datasets (given as arrays) and returns
   the correlation coefficient (R) and the p-value.
   
   --------------------------------------------
   Parameters:
        - array_1 (numpy.ndarray): Array of the first dataset to compare.
        - array_2 (numpy.ndarray): Array of the second dataset to compare.
        - test (str): Correlation test. By default, Pearson correlation test is selected.
        - alternative (str) = p_value test. By default, two-sided test is selected. 

    Returns:
        - The correlation coefficient (r) and p_value.
   --------------------------------------------
   """

   test.lower()
   if test == "spearman":
      R,p_value = stats.spearmanr(array_1, array_2, alternative=alternative)
      if verbose:
         print("Spearman correlation test was performed")
   
   elif test == "pearson":
      R,p_value = stats.pearsonr(array_1, array_2, alternative=alternative)
      if verbose:
         print("Spearman correlation test was performed")

   return R, p_value


def correlation_analysis_df(df, list_variables=list, fixed_variable=None):
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


def extract_correlations(x=str,y=str, correlation_table=None):
    """
    Extract the correlation coefficient and p-value of a given pair of variables 
    and a correlation table (this table can be obtained by using the "correlation_analysis_df" function).

    --------------------------------------------
    Parameters:
        - x,y (strings): variable names for which we want to extract the correlation coefficient and p-value.
        - correlation_table (pandas.DataFrame): Dataframe in which we have the correlation analysis and we want to extract the data.
    
    Returns:
        - R_p,pvalue_p,R_s,pvalue_s: Correlation coefficient and p-value of Pearson and Spearman tests for a given pair of variables.
    --------------------------------------------
    """

    # 1. Extract the correlation results for our variables x and y
    corr_results = correlation_table[
        ((correlation_table["array_1"] == x) & (correlation_table["array_2"] == y)) |
        ((correlation_table["array_1"] == y) & (correlation_table["array_2"] == x))
    ]

    # 2. Extract the correlation coefficient and p-value for Pearson and Spearman tests:
    R_p = float(corr_results["R_pearson"].values)
    pvalue_p = float(corr_results["pvalue_pearson"].values)
    R_s = float(corr_results["R_spearman"].values)
    pvalue_s = float(corr_results["pvalue_spearman"].values)

    return R_p,pvalue_p,R_s,pvalue_s


def extract_correlation_df(df_input, analysis=str, var1=str, var2=str, fixed_variable="fitness", columns=list, prot_sequence=str):
    """
    Extract the correlation coefficiens and p-value of a given pair of variables from an input dataframe.

    --------------------------------------------
    Parameters:
        - df_input (pandas.DataFrame): Dataframe from which we want to extract the data and perform the correlation analysis.
        - analysis (str): Type of analysis. Choices: "Pos", "Mut" or "WT".
        - var1, var2 (str): Variables to compare.
        - fixed_variable (str, optional): If provided, fix this variable and only do combinations with it. By default, "fitness" is selected.
        - columns(list): columns to perform correlation analysis.
        - prot_sequence (str): Protein sequence of the dataframe.
    Returns:
        - A dataframe with the correlation coefficients and p-values of the given pair of variables.
    ------------------------------------------
    """    
    
    # 1. According to the selected test...
    dict_analysis = {}
    # In case of position analysis:
    if df_input[analysis].dtype == "int64":
        seq_length = len(prot_sequence)

        for pos in range(0,seq_length):
            dict_analysis[f"df_{pos}_{analysis}"] = df_input[df_input[analysis] == pos]
    # Other way, mutation or wildtype analysis are selected
    else:
        for aa in amino_acids:
            dict_analysis[f"df_{aa}_{analysis}"] = df_input[df_input[analysis] == aa]
  

    # 2. Extract the correlation results and store them in a dataframe
    dict_corr_analysis = {}
    # Calculate the correlation of all these dataframes
    for df_id, df_content in dict_analysis.items():
        dict_corr_analysis[f"{df_id}"] = correlation_analysis_df(df_content, columns, fixed_variable=fixed_variable)

    # 3. Iterate through the dictionary, filter rows, and append the key as a new column
    df_output = pd.DataFrame()

    for key, df in dict_corr_analysis.items():
        filtered_rows = df.loc[(df["array_1"] == var1) & (df["array_2"] == var2)].copy()
        filtered_rows["dict_key"] = key  # Add a new column with the dictionary key
        df_output = pd.concat([df_output, filtered_rows], ignore_index=True)

    df_output = df_output.reset_index(drop=True)

    return df_output


def multiple_correlation_analysis(df, var1=str, var2=str,
                                  prot_sequence=str, analysis=str,
                                  fixed_variable=None, figsize=tuple, title = str):
    """
    Perform a multiple correlation analysis between two variables depending on the type of analysis selected (Pos, Mut or WT).
    This function creates a scatter plot and calculates the correlation coefficient and p-value.

    --------------------------------------------
    Parameters:
        - df (pandas.DataFrame): Dataframe from which we want to extract the data and perform the correlation analysis.
        - var1, var2 (str): Variables to compare.
        - prot_sequence (str): Protein sequence of the dataframe.
        - analysis (str): Variable to analysis. It can be "Pos", "Mut" or "WT".
        - fixed_variable (str, optional): If provided, fix this variable and only do combinations with it.
        - figsize (tuple, optional): Size of the figure. By default, (15, 15) is selected.
        - title (str, optional): Title of the figure. By default, "Correlation analysis" is selected.

    Returns:
        - A scatter plot with the correlation coefficients and p-values of the given pair of variables.
    --------------------------------------------
    """
    # 1. According to the selected test...
    dict_analysis = {}
    dict_corr_analysis = {}
    columns = [var1,var2]
    
    # In case of position analysis:
    if df[analysis].dtype == "int64":
        seq_length = len(prot_sequence)

        if seq_length%2 == 0:
            n_rows = seq_length//4
        else:
            n_rows = seq_length//4 + 1

        for pos in range(0,seq_length):
            dict_analysis[f"df_{pos}_{analysis}"] = df[df[analysis] == pos]

    # Other way, mutation or wildtype analysis are selected
    else:
        n_rows=5
        for aa in amino_acids:
            dict_analysis[f"df_{aa}_{analysis}"] = df[df[analysis] == aa]

    # 2. Then, calculate the correlation of all these dataframes
    for df_id, df_content in dict_analysis.items():
        dict_corr_analysis[f"{df_id}"] = correlation_analysis_df(df_content, columns, fixed_variable=fixed_variable)

    
    fig, axs = plt.subplots(n_rows, 4, figsize=figsize)
    axs = axs.flatten()
    
    for idx, (df_id, df_content) in enumerate(dict_analysis.items()):

        # Extract the correlation results for our variables fitness and global_value
        corr_results = dict_corr_analysis[df_id][
            (dict_corr_analysis[df_id]["array_1"] == var1) & (dict_corr_analysis[df_id]["array_2"] == var2) |
            ((dict_corr_analysis[df_id]["array_1"] == var2) & (dict_corr_analysis[df_id]["array_2"] == var1))
        ][['R_pearson', 'pvalue_pearson', 'R_spearman', 'pvalue_spearman']]


            # Extract the values
        R_p = float(corr_results["R_pearson"].values)
        pvalue_p = float(corr_results["pvalue_pearson"].values)
        R_s = float(corr_results["R_spearman"].values)
        pvalue_s = float(corr_results["pvalue_spearman"].values)

        # Scatter plot
        axs[idx].scatter(y=df_content[var1], x=df_content[var2], s=5)
        axs[idx].set_title(df_id)
        axs[idx].set_ylabel(var1)
        axs[idx].set_xlabel(var2)
        axs[idx].text(0.7, 0.5, f"R_Pearson = {R_p:.2f}; pvalue_p = {pvalue_p:.1e}", horizontalalignment='center', 
                      verticalalignment='center', transform=axs[idx].transAxes, color="blue")
        axs[idx].text(0.7, 0.4, f"R_Spearman = {R_s:.2f}; pvalue_s = {pvalue_s:.1e}", horizontalalignment='center', 
                      verticalalignment='center', transform=axs[idx].transAxes, color="red")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()