from SigProfilerExtractor import sigpro as sig
def extract_all():
    sig.sigProfilerExtractor("matrix", "results_combined", "data/combined.txt", reference_genome="GRCh37", 
                            minimum_signatures=1, maximum_signatures=20, nmf_replicates=100, cpu=-1)
    
if __name__ == "__main__":
    extract_all()