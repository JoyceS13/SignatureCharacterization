from SigProfilerExtractor import sigpro as sig
def stomach_cancer():
    sig.sigProfilerExtractor("matrix", "results_STAD", "data/STAD.txt", reference_genome="GRCh37", 
                            minimum_signatures=1, maximum_signatures=8, nmf_replicates=100, cpu=-1)
    
if __name__ == "__main__":
    stomach_cancer()