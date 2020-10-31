from model.noise_ddm import noise_ddm
from data_handling.analyzer import analyzer
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from time import time
import numpy as np
from itertools import product
import os

def main():
    os.chdir(r'C:\Users\Kamp\Documents\repos\DDM')
    # Define noise range and colors and init the analyzer
    noise_range = np.linspace(0.01,0.025,2)
    noise_colors = ['blue','white', 'pink']
    a = analyzer()
    
    # Run Simulations
    with PdfPages(a.pdf('Noise-DDM_Simulation.pdf')) as pdf:
        for level, color in product(noise_range, noise_colors):
            # Init model and simulate 1 second
            start = time()
            model = noise_ddm(drift=0.001, bias=0.5, noise_color=color, noise_level=level)
            model.simulate(n_samples=1000, time=1000)
            end = time()
            print('Simulation Time: ', end-start)
            
            # Plot and save to pdf file
            a.load(model)
            fig = a.plot_combined(n_trials=50)
            pdf.savefig(fig)
    
    # Plot stats
    with PdfPages(a.pdf('Noise-DDM_Stats.pdf')) as pdf: 
        fig = a.plot_stats()
        pdf.savefig(fig)

if __name__ == "__main__":
    main()