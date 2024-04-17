import pandas as pd
import numpy as np

def load_dna():
    dna = {}
    for i in range(24):
        if i == 22:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa', 'r')
        elif i == 23:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa', 'r')
        else:
            f = open('data/dna/Homo_sapiens.GRCh37.dna.chromosome.{}.fa'.format(i+1), 'r')
        content = f.read()
        f.close()
        seq = ''.join(content.split()[4:])
        if i ==22:
            dna['X'] = seq
        elif i == 23:
            dna['Y'] = seq
        else:
            dna[str(i+1)] = seq
    #print(len(dna[1]))
    return dna

def complement(base):
    if (base == 'A'):
        return 'T'
    elif( base == 'T'):
        return 'A'
    elif(base == 'C'):
        return 'G'
    elif(base == 'G'):
        return 'C'
    else:
        return 'N'
    

def log_mutations(input_file, output_file):
    df_in = pd.read_csv(input_file, delimiter="\t")
    samples = df_in['sample'].unique()
    dna = load_dna()
    df_brca = pd.read_csv("data/21BRCA.txt", delimiter="\t")
    df_out = pd.DataFrame(np.zeros((len(df_brca), len(samples))),columns=samples)
    df_out.insert(0,'Mutation Types', df_brca['Mutation Types'])
    ref_mismatches = 0
    indels = 0
    for ii in range(1,df_in.shape[0]):
        row = df_in.iloc[ii]
        sample = row['sample']
        chrom = row['chr']
        index = row['start']
        if row['reference'] == '-' or row['alt'] == '-':
            indels += 1
        elif dna[chrom][index-1] != row['reference']:
            ref_mismatches += 1
        else:
            up = dna[chrom][index-2]
            down = dna[chrom][index]
            mut = '{}[{}>{}]{}'.format(up,row['reference'], row['alt'],down)
            if mut not in df_out['Mutation Types'].values:
                mut = '{}[{}>{}]{}'.format(complement(down),complement(row['reference']), complement(row['alt']),complement(up))
            df_out.loc[df_out['Mutation Types'] == mut, sample] += 1
    df_out.to_csv(output_file, sep="\t", index=False)
    print("Indels: ", indels)
    print("Reference mismatches: ", ref_mismatches)
    
def concatenate_files(files, output_file):
    df = pd.read_csv(files[0], delimiter="\t")
    df.set_index('Mutation Types', inplace=True)
    for i in range(1, len(files)):
        df = pd.concat([df, pd.read_csv(files[i], delimiter="\t").set_index('Mutation Types')], axis=1, join='outer', sort=False)
    df.to_csv(output_file, sep="\t", index=True)
    
if __name__=="__main__":
    # input_skcm = "C:/Users/wyjsu/Downloads/mc3_SKCM_mc3.txt/SKCM_mc3.txt"
    # output_skcm = "data/SKCM.txt"
    # with open(input_skcm, 'r') as f:
    #     total_num_mutations_skcm = sum(1 for line in f)
    # print(total_num_mutations_skcm)
    # #log_mutations(input_skcm, output_skcm)
    # input_brca = "C:/Users/wyjsu/Downloads/mc3_BRCA_mc3.txt/BRCA_mc3.txt"
    # output_brca = "data/BRCA.txt"
    # with open(input_brca, 'r') as f:
    #     total_num_mutations_brca = sum(1 for line in f)
    # print(total_num_mutations_brca)
    # #log_mutations(input_brca, output_brca)
    # input_lung = "C:/Users/wyjsu/Downloads/mc3_LUNG_mc3.txt/LUNG_mc3.txt"
    # output_lung = "data/LUNG.txt"
    # with open(input_lung, 'r') as f:
    #     total_num_mutations_lung = sum(1 for line in f)
    # print(total_num_mutations_lung)
    # #log_mutations(input_lung, output_lung)
    # input_stad = "C:/Users/wyjsu/Downloads/mc3_STAD_mc3.txt/STAD_mc3.txt"
    # output_stad = "data/STAD.txt"
    # with open(input_stad, 'r') as f:
    #     total_num_mutations_stad = sum(1 for line in f)
    # print(total_num_mutations_stad)
    # #log_mutations(input_stad, output_stad)
    
    concatenate_files(["data/SKCM.txt", "data/BRCA.txt", "data/LUNG.txt", "data/STAD.txt"], "data/combined.txt")
