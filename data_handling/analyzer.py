import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from time import time
import seaborn as sns
import os, random
from itertools import product
import pandas as pd
import numpy as np

class analyzer(): 
    def __init__(self, model=None):
        self.parentdir = r'C:\Users\Kamp\Documents\repos\DDM'
        self.plotdir = os.path.join(self.parentdir, 'plots')
        self.model = model
        self.rt_dict = {'Color':[], 'Noise_Level':[], 'Reaction_Time':[]}
        self.rt_sd_dict = {'Color':[], 'Noise_Level':[], 'SD':[]}
    
    def pdf(self, name):
        return os.path.join(self.plotdir, name)
    
    def load(self, model):
        self.model = model
        # Transform reaction time
        hits_rt = self.model.hits_rt.astype(int)
        self.hits_rt = np.repeat(np.arange(0, self.model.time), hits_rt)
        self.rt_sd = np.std(self.hits_rt)

        self.rt_dict['Reaction_Time'].extend(self.hits_rt.tolist())
        self.rt_dict['Color'].extend([self.model.noise_color.capitalize()]*len(self.hits_rt))
        self.rt_dict['Noise_Level'].extend([self.model.noise_level]*len(self.hits_rt))

        self.rt_sd_dict['SD'].append(self.rt_sd)
        self.rt_sd_dict['Color'].append(self.model.noise_color.capitalize())
        self.rt_sd_dict['Noise_Level'].append(self.model.noise_level)


    def plot_trajectory(self, n_trials, ax):
        # Set up boundaries
        for bound in 0, self.model.bound:
            ax.plot(np.repeat([bound], self.model.time), linewidth=0.5, color ='k')
        ax.plot(np.repeat([self.model.bias], self.model.time), 'k', linestyle=(0,(1,10)))
        # plot trajectories
        for trial in random.sample(self.model.trajectory.tolist(), n_trials):
            trial = np.array(trial)
            idx = np.diff(trial,prepend=1)!=0
            sns.lineplot(x = np.arange(0,len(trial[idx])), y=trial[idx],ax=ax, linewidth=0.2)
        # Configure ax
        ax.set_ylim(-0.1, self.model.bound+0.1)
        ax.set_yticks([0,self.model.bias, self.model.bound])
        ax.set_xlabel('Time [ms]')
        textstr = f'Example Trials = {n_trials}\nDrift = {self.model.drift}'
        ax.text(0.8, 0.85, textstr, transform=ax.transAxes, fontsize=15, linespacing=2,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='lightgray', alpha=0.8))
        
    def plot_rt(self, ax):
        # Plot hist plot of reaction time        
        sns.histplot(self.hits_rt, binwidth=10, ax=ax, alpha=0.5, kde=True, line_kws={'alpha':0.75})
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, nr: val/self.model.n_samples))
        ax.set_ylabel('Probability')
        textstr = f'# Samples = {self.model.n_samples}\nReaction Time SD = {np.round(self.rt_sd,1)}'
        ax.text(0.8, 0.85,textstr, transform=ax.transAxes, fontsize=15, linespacing=2,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='lightgray', alpha=0.8))
        ax.set_title('Reaction Time Distribution', fontsize=15)
    
    def plot_combined(self, n_trials):
        sns.set_style('white'); sns.set_palette('deep')
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3,1, hspace=0.0)
        ax = []
        ax.append(fig.add_subplot(gs[0,0]))
        ax.append(fig.add_subplot(gs[1:,0], sharex=ax[0]))
        self.plot_rt(ax[0])
        self.plot_trajectory(n_trials, ax[1])
        ax[0].set_title(f'{self.model.noise_color.capitalize()} Noise, SD = {self.model.noise_level}', fontsize=25, pad=20)
        return fig
    
    def plot_stats(self):
        """
        Boxplot of the reaction time over different noise colors
        """
        sns.set_style('whitegrid')
        df = pd.DataFrame(self.rt_dict)
        sd_df = pd.DataFrame(self.rt_sd_dict)
        # Get list of unique noise levels
        noise_levels = sorted(set(df['Noise_Level']))
        # Init figure
        fig, axes = plt.subplots(1,len(noise_levels), figsize=(6*len(noise_levels),6), sharey=True)
        # Get y text position
        ypos = np.max(df['Reaction_Time'])
        for level, ax in zip(noise_levels, axes):
            spl_df = df.loc[df['Noise_Level']==level]
            sns.boxplot(data=spl_df, x='Color', y='Reaction_Time', ax=ax, palette="vlag", saturation=1, fliersize=2)
            ax.set_xlabel('Noise Color', labelpad=15)
            ax.set_title(f'Noise SD {level}', pad=20)
            
            # Add SD values as text
            spl_sd = sd_df.loc[sd_df['Noise_Level']==level]
            sd_strings = [f'SD = {np.round(sd,1)}' for sd in spl_sd['SD']]
            for loc, sd in zip(ax.get_xticks(), sd_strings):
                ax.text(loc, ypos, sd, horizontalalignment='center', size='small', style='italic', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))  
        for ax in axes[1:]: ax.set_ylabel('')
        axes[0].set_ylabel('Reaction Time [ms]')
        sns.despine()

        return fig 
