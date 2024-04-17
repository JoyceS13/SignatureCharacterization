from SigProfilerExtractor import sigpro as sig
def breast_cancer():
    sig.sigProfilerExtractor("matrix", "results_21BRCA.txt", "data/BRCA.txt", reference_genome="GRCh37", 
                            minimum_signatures=1, maximum_signatures=10, nmf_replicates=100, cpu=-1)
    
if __name__ == "__main__":
    breast_cancer()