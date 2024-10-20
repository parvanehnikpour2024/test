# Main aim of this script:
# This script performs analyses on bulk RNA-seq data. 
# Initially, it merges two RNA-seq files from different sample runs. 
# After merging, it includes functionality for (optionally) renaming samples, 
# followed by two dimensionality reduction methods: PCA and UMAP.

##############################################################
#######################Importing packages#####################
##############################################################
# Once the packages are installed (using pip install -r requirements.txt), we import them into our .py script
import sys

# Print the Python executable being used
print("Python executable being used:", sys.executable)

# Print the Python version being used
print("Python version:", sys.version)

import pandas as pd                                   # Import pandas (a powerful tool for data manipulation and analysis) library and alias it as 'pd'
import seaborn as sns                                 # Import seaborn for statistical data visualization
import matplotlib.pyplot as plt                       # Import matplotlib for plotting data
import umap.umap_ as umap                             # Import UMAP for dimensionality reduction
from sklearn.decomposition import PCA                 # Import PCA for Principal Component Analysis
from sklearn.preprocessing import StandardScaler      # Import StandardScaler for feature scaling

##############################################################
########Data Preparation (Merging and Renaming Samples)#######
##############################################################
#Merging
#Opening Excel files (In this example, run1 and run2 of our RNASeq count data)
file1 = pd.read_excel("run1.xlsx")  # Make sure the file is in the same directory
file2 = pd.read_excel("run2.xlsx")  # Make sure the file is in the same directory

#Combining two Excel files (file1 and file2 of our RNASeq count data)
combined_file = pd.merge(file1, file2, on="gene_symbol", how="outer")  # Adjust the 'on' parameter based on your own data
combined_file.fillna(0, inplace=True)  # Replace NaN values with 0

#Checking for NA values in the merged (combined) file
if combined_file.isna().any().any():  # Check if any NA values exist in the DataFrame
    print("The file contains NA values.")
else:
    print("The file is NA-free and ready for saving as an xlsx output file.")

#Writing the combined data to a new Excel file
combined_file.to_excel("combined_run1run2.xlsx", index=False)  # Save the combined file as a new excel file
print("The xlsx output file: combined_run1run2.xlsx was created.")

# Renaming (An optional step just for learning how to rename if needed)
# Renaming specific columns as needed. Left side names are old ones and right side shows the new assigned name
rename_dict = {
    'IPS_BD_RA_1': 'IPS_BD_RA_1_new',       # Map old column name to new column name
    'IPS_BD_RA_2': 'IPS_BD_RA_2_new',       # Map old column name to new column name
    # Add more mappings as needed
}

combined_file.rename(columns=rename_dict, inplace=True) # Rename the columns in the combined_file DataFrame using the rename_dict

# Display the first few rows of the renamed DataFrame
print(combined_file.head())  

# Writing (saving) the renamed data to a new output Excel file
combined_file.to_excel("renamed_combined_run1run2.xlsx", index=False)
print("The xlsx output file: renamed_combined_run1run2.xlsx was created.")

##############################################################
#############PCA (Principal Component Analysis)###############
##############################################################
# Info: PCA (Principal Component Analysis) is a widely used dimensionality reduction technique in biology, especially in genomics and transcriptomics, for simplifying high-dimensional data while retaining as much of the original variability as possible. In biological studies, such as gene expression analysis, PCA helps to explore patterns, identify relationships, and reduce noise in complex datasets.

# Performing PCA
# Load the metadata file
samples_metadata = pd.read_excel("Samples_metadata.xlsx", sheet_name="Sheet1")
print(samples_metadata.head(10))  # Display the first 10 rows of sample metadata

# Prepare the RNA-seq data for PCA
RNAseq1 = combined_file.set_index('gene_symbol')  # Set 'gene_symbol' as index
RNAseq1 = RNAseq1.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
RNAseq1_t = RNAseq1.T  # Transpose the DataFrame (genes become columns, samples become rows)

# Performing PCA
scaler = StandardScaler()
RNAseq1_scaled = scaler.fit_transform(RNAseq1_t)  # Scale the data
print("Scaled RNAseq data shape:", RNAseq1_scaled.shape)  # Print shape of scaled data

pca = PCA()
principal_components = pca.fit_transform(RNAseq1_scaled)  # Perform PCA

# Print number of components and explained variance ratio
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Create a DataFrame for the first two principal components
PCX = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])
PCX['Sample_IDs'] = samples_metadata.iloc[:, 0].values  # Add a new column with sample names and Ensure proper alignment with sample IDs

# Merge the PCA result with sample metadata
samples_metadata.set_index('Sample_IDs', inplace=True)  # Set index for merging
PCX2 = PCX.merge(samples_metadata, on='Sample_IDs', how='left')  # Merge data

# Check merged PCA data and column names
print(PCX2.head())  # Display the merged data
print(PCX2.info())  # Display info about merged DataFrame

# Final verification of sample names
print("Merged Sample IDs:", PCX2['Sample_IDs'].unique())  # Print unique Sample IDs in the merged DataFrame

# PCA visualization
# Define custom colors and shapes for the plot
colors = {"BD": "#FF0000", "CNTRL": "#0000FF"}
shapes = {"IPS": "o", "NPC": "s", "NRN": "D"}

# Plotting the PCA result
plt.figure(figsize=(10, 8))

# Create scatterplot
sns.scatterplot(x='PC1', y='PC2', 
                hue='Disease_type', 
                style='Differentiation_type', 
                data=PCX2, 
                palette=colors, 
                markers=shapes, 
                s=100)

# Adding sample labels above the points for RNAseqrun
for i, row in PCX2.iterrows():  # Use iterrows() for better readability
    plt.text(row['PC1'], row['PC2'], 
             str(row['RNAseqrun']), 
             fontsize=9,  # Adjustable font size
             ha='center',  # Horizontal alignment
             va='bottom',  # Vertical alignment
             color='black')  # Color of the text

# Customizing the plot
plt.title('PCA plot of RNA-seq Data (PC1 vs PC2)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.grid(True)

# Prepare handles and labels for the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Custom labels for RNAseq run, directly using existing handles if applicable
rna_run_labels = ['1: RNAseq Run 1', '2: RNAseq Run 2']
rna_run_handles = [plt.Line2D([0], [0], marker='None', color='k', label=label) for label in rna_run_labels]

# Add custom handles for RNAseq runs to the existing handles
handles += rna_run_handles

# Recreate the legend with the updated handles and labels
plt.legend(handles=handles, labels=labels + rna_run_labels, 
           title="Disease Type, Differentiation Type, and RNAseq Run", 
           loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Save the plot as JPEG
plt.savefig("PCA_plot.jpeg", format='jpeg', bbox_inches='tight')  # Optional saving step

# Adjust layout and show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

##############################################################
#####UMAP (Uniform Manifold Approximation and Projection)#####
##############################################################
# Info: UMAP is a popular dimensionality reduction technique used to visualize high-dimensional data, such as gene expression data from RNA-seq experiments. It is particularly useful for simplifying and interpreting complex biological datasets, often revealing patterns and relationships that might not be apparent in the raw data.

# Performing UMAP
# Create UMAP Object and Perform Transformation
umap_obj = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = umap_obj.fit_transform(RNAseq1_scaled)

# Create a DataFrame for the UMAP embedding with sample identifiers
umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['Sample_IDs'] = RNAseq1_t.index  # Adding sample identifiers

# Check the newly created UMAP DataFrame
print(umap_df.head())  # Display first few rows

# Merge the UMAP result with sample metadata
umap_merged = umap_df.merge(samples_metadata, on='Sample_IDs', how='left')
print(umap_merged.head())
print("UMAP embedding shape:", embedding.shape)  # Check the dimensions

# UMAP visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', 
                hue='Disease_type', 
                style='Differentiation_type', 
                data=umap_merged, 
                palette=colors, 
                markers=shapes, 
                s=100)

# Adding sample labels above the points for RNAseq run
for i, row in umap_merged.iterrows():  # Use iterrows() for clarity
    plt.text(row['UMAP1'], row['UMAP2'], 
             str(row['RNAseqrun']), 
             fontsize=9, ha='center', va='bottom', color='black')

# Customizing the plot
plt.title('UMAP plot of RNA-seq Data', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.grid(True)

# Prepare handles and labels for the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Custom labels for RNAseq run
rna_run_labels = ['1: RNAseq Run 1', '2: RNAseq Run 2']
rna_run_handles = [plt.Line2D([0], [0], marker='None', color='k', label=label) for label in rna_run_labels]

# Add custom handles for RNAseq runs to the existing handles
handles += rna_run_handles

# Recreate the legend with the updated handles and labels
plt.legend(handles=handles, labels=labels + rna_run_labels, 
           title="Disease Type, Differentiation Type, and RNAseq Run", 
           loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Save the plot
plt.savefig("UMAP_plot.jpeg", format='jpeg', bbox_inches='tight')

# Adjust layout and show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()