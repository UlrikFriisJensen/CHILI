# %% Imports
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mode

warnings.simplefilter(action="ignore")


# %% Functions
class custom_float_aggfunc():
    def __init__(self, fstring_format=".3f"):
        self.fstring_format = fstring_format
    
    def __call__(self, x):
        if x.std() > 0.00001:
            return rf"${x.mean():{self.fstring_format}} \pm {x.std():{self.fstring_format}}$"
        else:
            return rf"${x.mean():{self.fstring_format}}$"

def custom_int_aggfunc(x):
    return f"${mode(x).mode:.0f}$"

def main(results_dir, fstring_format=".3f"):
    # Initialize an empty list to store the dataframes
    dfs = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(results_dir):
        # Iterate over the files in the current directory
        for file in files:
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Construct the full path to the CSV file
                file_path = os.path.join(root, file)

                # Read the CSV file into a dataframe
                df = pd.read_csv(file_path)

                # Append the dataframe to the list
                dfs.append(df)

    # Concatenate all the dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True).drop("Unnamed: 0", axis=1)

    # Split the dataframe into two separate dataframes
    property_df = combined_df[
        ~combined_df["Task"].str.endswith(("SAXS", "xPDF", "XRD"))
    ]
    structure_df = combined_df[
        combined_df["Task"].str.endswith(("SAXS", "xPDF", "XRD"))
    ]

    ### Property dataframe
    # Convert the 'Dataset' and 'Model' columns to categorical data types
    property_df["Dataset"] = property_df["Dataset"].astype(
        pd.api.types.CategoricalDtype(categories=["CHILI-3K", "CHILI-100K"])
    )
    property_df["Model"] = property_df["Model"].astype(
        pd.api.types.CategoricalDtype(
            categories=[
                "RandomClass",
                "MostFrequentClass",
                "Mean",
                "GCN",
                "GraphSAGE",
                "GAT",
                "GIN",
                "EdgeCNN",
                "GraphUNet",
                "PMLP",
            ]
        )
    )

    # Specify the order of the rows and columns
    column_order = [
        "AtomClassification",
        "CrystalSystemClassification",
        "SpacegroupClassification",
        "PositionRegression",
        "DistanceRegression",
        "SAXSRegression",
        "XRDRegression",
        "xPDFRegression",
    ]
    # Row order from least to most trainable parameters
    # row_order = ['RandomClass', 'MostFrequentClass', 'Mean', 'GCN', 'PMLP', 'GraphSAGE', 'GAT', 'GraphUNet', 'GIN', 'EdgeCNN'] # TODO: Can't figure out how to sort rows in double indexed pivot table

    # Create and save the LaTeX table
    property_df.pivot_table(
        index=["Dataset", "Model"],
        columns="Task",
        values="Test metric",
        aggfunc=custom_float_aggfunc(fstring_format=fstring_format),
    ).reindex(column_order, axis=1, fill_value="---").to_latex(
        f"./{results_dir}/propertyPredictionTasksTable.tex", column_format="lccccccccc"
    )

    # Create and save LaTeX table with model parameters
    row_order = ["GCN", "PMLP", "GraphSAGE", "GAT", "GraphUNet", "GIN", "EdgeCNN"]
    property_df.pivot_table(
        index=["Model"],
        values="Trainable parameters",
        aggfunc=custom_int_aggfunc,
        fill_value=0,
    ).reindex(row_order, axis=0, fill_value="---").to_latex(
        f"./{results_dir}/ModelParametersTable.tex", column_format="lc"
    )
    ### Structure dataframe
    # Convert the 'Dataset' column to categorical data types
    structure_df["Dataset"] = structure_df["Dataset"].astype(
        pd.api.types.CategoricalDtype(categories=["CHILI-3K", "CHILI-100K"])
    )

    # Specify the order of the columns
    column_order = [
        "CrystalSystemClassificationSAXS",
        "CrystalSystemClassificationXRD",
        "CrystalSystemClassificationxPDF",
        "SpacegroupClassificationSAXS",
        "SpacegroupClassificationXRD",
        "SpacegroupClassificationxPDF",
        "CellParamsRegressionSAXS",
        "CellParamsRegressionXRD",
        "CellParamsRegressionxPDF",
        "UnitCellPositionRegressionSAXS",
        "UnitCellPositionRegressionXRD",
        "UnitCellPositionRegressionxPDF",
        "AbsPositionRegressionSAXS",
        "AbsPositionRegressionXRD",
        "AbsPositionRegressionxPDF",
    ]

    # Create and save the LaTeX table
    structure_df.pivot_table(
        index=["Task"], 
        columns="Dataset", 
        values="Test metric", 
        aggfunc=custom_float_aggfunc(fstring_format=fstring_format),
    ).reindex(column_order, axis=0, fill_value="---").to_latex(
        f"./{results_dir}/structurePredictionTasksTable.tex", column_format="lcc"
    )


# %% Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--fstring_format", default=".3f", type=str)
    args = vars(parser.parse_args())
    
    main(args["results_dir"], args["fstring_format"])
