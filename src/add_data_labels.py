import pandas as pd

def load_data_files():
    skcm_data = pd.read_csv("data/SKCM.txt", delimiter="\t")
    brca_data = pd.read_csv("data/BRCA.txt", delimiter="\t")
    lung_data = pd.read_csv("data/LUNG.txt", delimiter="\t")
    stad_data = pd.read_csv("data/STAD.txt", delimiter="\t")
    return skcm_data, brca_data, lung_data, stad_data

def add_data_labels(activity_file, output_file, normalized=True):
    activity_data = pd.read_csv(activity_file, delimiter="\t", index_col=0)
    skcm_data, brca_data, lung_data, stad_data = load_data_files()
    data_labels = []
    for i in range(len(activity_data)):
        row = activity_data.iloc[i]
        sample = row.name
        if sample in skcm_data.columns:
            data_labels.append('SKCM')
            total_num_mutations = sum(skcm_data[sample])
        elif sample in brca_data.columns:
            data_labels.append('BRCA')
            total_num_mutations = sum(brca_data[sample])
        elif sample in lung_data.columns:
            data_labels.append('LUNG')
            total_num_mutations = sum(lung_data[sample])
        elif sample in stad_data.columns:
            data_labels.append('STAD')
            total_num_mutations = sum(stad_data[sample])
        else:
            raise ValueError("Sample not found in any data file")
        
        if normalized:
            activity_data.iloc[i] = row / total_num_mutations
    
    activity_data.insert(0, 'Data Label', data_labels)
    activity_data.to_csv(output_file, sep="\t", index=True)
    
def median_activity_per_cancer_type():
    activities = pd.read_csv("data/labeled_data_11.txt", delimiter="\t", index_col=0)
    for cancer_type in activities['Data Label'].unique():
        cancer_type_data = activities[activities['Data Label'] == cancer_type]
        median_activity = cancer_type_data.iloc[:, 1:].median()
        average_activity = cancer_type_data.iloc[:, 1:].mean()
        #print all results to file
        with open("results_combined/Median_Activities.txt", 'a') as f:
            f.write(cancer_type + "\n")
            f.write("Median Activity\n")
            f.write(str(median_activity) + "\n\n")
            f.write("Average Activity\n")
            f.write(str(average_activity) + "\n\n")

    
if __name__=="__main__":
    # activity_file = "results_combined/SBS96_11_Signatures/Activities/SBS96_S11_NMF_Activities.txt"
    # output_file = "data/labeled_data_11.txt"
    # add_data_labels(activity_file, output_file)
    median_activity_per_cancer_type()
    