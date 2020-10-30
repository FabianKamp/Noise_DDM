import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from time import time
import seaborn as sns
from noise.noise import colored_noise
import os, random
from itertools import product

class noise_ddm(object):
    def __init__(self, drift=0.001, bound=1, bias=0.5, noise_level=0.01, noise_color='white'):
        """
        Set parameters 
        """
        self.drift = drift
        self.bound = bound
        self.bias = bias
        self.noise_level = noise_level
        self.noise_color = noise_color
        self.trajectory = np.array([])
        self.rt = np.array([])
        self.n_samples = 0
        self.time = 1000
    
    def simulate(self, time=1000, n_samples=1000):
        for param in [self.drift, self.bound, self.bias, self.noise_level, self.noise_color]:
            assert not isinstance(param, list), 'Parameter range must be fitted before starting simulation.'
        # Set up attributes
        self.time=time
        self.n_samples=n_samples
        self.rt = np.zeros(time)
        self.trajectory = np.zeros((n_samples,time))
        self.noise = colored_noise(self.noise_color, low_cut=5, high_cut=495, time=time, n_samples=n_samples)

        # Set up trace, trajectory and rt array
        trace = np.repeat(self.bias,self.n_samples)
        trajectory = self.trajectory
        trajectory[:,0] = trace
        rt =  self.rt
        # Set up indexing
        prev_crossed = np.zeros(self.n_samples, dtype='bool')
        out = np.zeros(self.n_samples,dtype='bool')
        for t in np.arange(1,self.time):
            # update trace
            trace[out==False] += self.drift + self.noise[:n_samples,t]*self.noise_level
            # get reaction time
            curr_crossed = np.sum([trace>=self.bound]) - np.sum(prev_crossed)
            rt[t]=curr_crossed
            # update traces that crossed 
            prev_crossed[trace>=self.bound]=True
            out[trace>=self.bound] = True; out[trace<=0]=True
            n_samples = np.sum(out==False)
            # save trace to final trajectory
            trace[trace>=self.bound] = self.bound; trace[trace<=0]=0
            trajectory[:,t] = trace
            # Return if all traces are out
            if np.all(out):
                self.trajectory=self.trajectory[:,:t+1]
                return self.rt, self.trajectory
        return self.rt, self.trajectory
    
    def plot_trajectory(self, n_trials, ax):
        # Set up boundaries
        for bound in 0, self.bound:
            ax.plot(np.repeat([bound], self.time), linewidth=0.5, color ='k')
        ax.plot(np.repeat([self.bias], self.time), 'k', linestyle=(0,(1,10)))
        # plot trajectories
        for trial in random.sample(self.trajectory.tolist(), n_trials):
            trial = np.array(trial)
            idx = np.diff(trial,prepend=1)!=0
            sns.lineplot(x = np.arange(0,len(trial[idx])), y=trial[idx],ax=ax, linewidth=0.5)
        # Configure ax
        ax.set_ylim(-0.1, self.bound+0.1)
        ax.set_yticks([0,self.bias, self.bound])
        ax.set_xlabel('Time [ms]')
        textstr = f'Ex. Trials = {n_trials}\nDrift = {self.drift}\nNoise Color = {self.noise_color.capitalize()}\nNoise STD = {self.noise_level}'
        ax.text(0.8, 0.85, textstr, transform=ax.transAxes, fontsize=15, linespacing=1.5,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='darkgray', alpha=0.8))
        
    def plot_rt(self, ax):
        # Plot reaction times as bar plot
        r = self.rt.astype(int); 
        t = np.arange(0,self.time)
        h = np.repeat(np.arange(0, self.time), r)
        sns.histplot(h, binwidth=10, ax=ax, alpha=0.5, kde=True, line_kws={'alpha':0.75})
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, nr: val/self.n_samples))
        ax.set_ylabel('Probability')
        textstr = f'# Samples = {self.n_samples}'
        ax.text(0.8, 0.85,textstr, transform=ax.transAxes, fontsize=15,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='darkgray', alpha=0.8))
        ax.set_title('Reaction Time Distribution', fontsize=15)
        #sns.barplot(x = np.arange(0, self.time), y = self.rt, ax=ax, alpha=0.75, color='tab:blue')
    
    def plot_combined(self, n_trials):
        sns.set_style('white'); sns.set_palette('deep')
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3,1, hspace=0.0)
        ax = []
        ax.append(fig.add_subplot(gs[0,0]))
        ax.append(fig.add_subplot(gs[1:,0]))
        self.plot_rt(ax[0])
        self.plot_trajectory(n_trials, ax[1])
        ax[0].set_xticks([])
        return fig

if __name__ == "__main__":
    os.chdir(r'C:\Users\Kamp\Documents\repos\DDM')
    nr_noise_levels = 2
    with PdfPages('Noise_DDM.pdf') as pdf:
        for level, color in product(np.linspace(0.01,0.015,nr_noise_levels), ['white', 'blue', 'pink']):
            m = noise_ddm(drift=0.002, bias=0.5, noise_color=color, noise_level=level)
            m.simulate(n_samples=1000, time=1000)
            fig = m.plot_combined(n_trials=50)
            pdf.savefig()
