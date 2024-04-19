import numpy as np 
import pandas as pd
from sklearn.manifold import MDS 
import matplotlib.pyplot as plt

def cosine_similarity(signature1, signature2):
    """
    Compute the cosine similarity between two signatures.
    """
    return np.dot(signature1, signature2) / (np.linalg.norm(signature1) * np.linalg.norm(signature2))

def load_signatures(file_path, data_name):
    """
    Load signatures from a file.
    """
    df = pd.read_csv(file_path, sep="\t", index_col=0)
    signatures = {}
    for i in range(df.shape[1]):
        column = df.iloc[1:, i]
        signatures[data_name + "_" + df.columns[i]] = np.array(column.to_list())
    return signatures

def compare_signatures(signatures1, signatures2):
    
    similarities = []
    for signature1 in signatures1:
        for signature2 in signatures2:
            similarity = cosine_similarity(signatures1[signature1], signatures2[signature2])
            similarities.append((signature1, signature2, similarity))
            
    return similarities

def get_high_similarity_pairs(similarities, threshold=0.8):
    """
    Get pairs of signatures with similarity above a threshold.
    """
    high_similarity_pairs = []
    for similarity in similarities:
        if similarity[2] > threshold:
            high_similarity_pairs.append(similarity)
            
    return high_similarity_pairs

def one_vs_all(file_path, comparison_files):
    """
    Compare one signature file against all signatures in another file.
    """
    signatures = load_signatures(file_path, "ALL ")
    all_similarities = []
    names = ["BRCA", "LUNG", "SKCM", "STAD"]
    for ii,comparison_file in enumerate(comparison_files):
        other_signatures = load_signatures(comparison_file, names[ii])
        similarities = compare_signatures(signatures, other_signatures)
        all_similarities.extend(similarities)
        
    high_similarity_pairs = get_high_similarity_pairs(all_similarities)
    number_of_similar_signatures = len(set([pair[0] for pair in high_similarity_pairs]))
    return number_of_similar_signatures, high_similarity_pairs

def get_mds_coordinates(signatures):
    """
    Get MDS coordinates of the signatures.
    """
    mds = MDS(n_components=2, dissimilarity="precomputed")
    distances = np.zeros((len(signatures), len(signatures)))
    for i, signature1 in enumerate(signatures):
        for j, signature2 in enumerate(signatures):
            distances[i, j] = 1 - cosine_similarity(signatures[signature1], signatures[signature2])
    coordinates = mds.fit_transform(distances)
    print(mds.stress_)
    return coordinates

def plot_mds_coordinates(signatures, coordinates):
    """
    Plot MDS coordinates of the signatures and return the figure.
    """
    fig, ax = plt.subplots()
    #ax.scatter(coordinates[:, 0], coordinates[:, 1])
    #color points by class, annotate signature names
    for i, signature in enumerate(signatures):
        if "ALL" in signature:
            ax.plot(coordinates[i, 0], coordinates[i, 1], "o", color="red")
        elif "BRCA" in signature:
            ax.plot(coordinates[i, 0], coordinates[i, 1], "o", color="blue")
        elif "LUNG" in signature:
            ax.plot(coordinates[i, 0], coordinates[i, 1], "o", color="green")
        elif "SKCM" in signature:
            ax.plot(coordinates[i, 0], coordinates[i, 1], "o", color="purple")
        elif "STAD" in signature:
            ax.plot(coordinates[i, 0], coordinates[i, 1], "o", color="orange")
        ax.text(coordinates[i, 0], coordinates[i, 1], signature[-1])
    
    #add labels
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    #plot legend
    ax.plot([], [], "o", color="red", label="ALL")
    ax.plot([], [], "o", color="blue", label="BRCA")
    ax.plot([], [], "o", color="green", label="LUNG")
    ax.plot([], [], "o", color="purple", label="SKCM")
    ax.plot([], [], "o", color="orange", label="STAD")
    ax.legend()

    return fig
    
def MDS_3D(signatures):
    """
    Get MDS coordinates of the signatures.
    """
    mds = MDS(n_components=3, dissimilarity="precomputed")
    distances = np.zeros((len(signatures), len(signatures)))
    for i, signature1 in enumerate(signatures):
        for j, signature2 in enumerate(signatures):
            distances[i, j] = 1 - cosine_similarity(signatures[signature1], signatures[signature2])
    coordinates = mds.fit_transform(distances)
    print(mds.stress_)
    
    #plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #color points by class, annotate signature names
    for i, signature in enumerate(signatures):
        if "ALL" in signature:
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color="red")
        elif "BRCA" in signature:
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color="blue")
        elif "LUNG" in signature:
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color="green")
        elif "SKCM" in signature:
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color="purple")
        elif "STAD" in signature:
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color="orange")
        ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], signature[-1])
        
    #plot legend
    ax.plot([], [], "o", color="red", label="ALL")
    ax.plot([], [], "o", color="blue", label="BRCA")
    ax.plot([], [], "o", color="green", label="LUNG")
    ax.plot([], [], "o", color="purple", label="SKCM")
    ax.plot([], [], "o", color="orange", label="STAD")
    ax.legend()    
        
    return fig
    
def main_MDS(input_files, output_file):
    """
    Main function.
    """
    names = ["ALL ","BRCA", "LUNG", "SKCM", "STAD"]
    signatures = {}
    for input_file in input_files:
        signatures.update(load_signatures(input_file, names[input_files.index(input_file)]))
    coordinates = get_mds_coordinates(signatures)
    fig = plot_mds_coordinates(signatures, coordinates)
    #save to file
    fig.savefig(output_file)
    
def main_MDS_3D(input_files, output_file):
    """
    Main function.
    """
    names = ["ALL ","BRCA", "LUNG", "SKCM", "STAD"]
    signatures = {}
    for input_file in input_files:
        signatures.update(load_signatures(input_file, names[input_files.index(input_file)]))
    fig = MDS_3D(signatures)
    #save to file
    fig.savefig(output_file)
    

def main(input_file, comparison_files, output_file):
    """
    Main function.
    """
    number_of_similar_signatures, high_similarity_pairs = one_vs_all(input_file, comparison_files)
    with open(output_file, "w") as f:
        f.write("Number of similar signatures: " + str(number_of_similar_signatures) + "\n")
        for pair in high_similarity_pairs:
            f.write(pair[0] + "\t" + pair[1] + "\t" + str(pair[2]) +"\n")
            
if __name__ == "__main__":
    combined_results = "results_combined/SBS96_11_Signatures/Signatures/SBS96_S11_Signatures.txt"
    results_individual = ["results_{}/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Signatures/SBS96_De-Novo_Signatures.txt".format(data) for data in ["BRCA", "LUNG", "SKCM", "STAD"]]
    # main(combined_results, 
    #      results_individual, 
    #      "results_combined/SBS96_S11_Signatures_Similarities_08.txt")
    #main_MDS([combined_results] + results_individual, "results_combined/SBS96_S11_Signatures_MDS.png")
    main_MDS_3D([combined_results] + results_individual, "results_combined/SBS96_S11_Signatures_MDS_3D.png")
    

