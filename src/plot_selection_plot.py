from SigProfilerExtractor import estimate_best_solution as ebs

def plot_selection_plot(mutation_data, sol_folder, out_folder):
    ebs.estimate_solution(base_csvfile=sol_folder + "/All_solutions_stat.csv", 
            All_solution=sol_folder, 
            genomes=mutation_data, 
            output=out_folder, 
            title="",
            stability=0.6, 
            min_stability=0, 
            combined_stability=0.6,
            allow_stability_drop=True,
            exome=False)
    
if __name__ == "__main__":
    plot_selection_plot("data/combined.txt", "results_combined", "selections_plots/test4")
    
    