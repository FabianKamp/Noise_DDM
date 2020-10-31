import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from time import time
import seaborn as sns
from colored_noise.noise import colored_noise
import os, random
from itertools import product
from time import time
from multiprocessing import Pool

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
        self.time = 1000
        self.n_samples = 0
  
    def simulate(self, time=1000, n_samples=1000, chunk_size = 1000):
        assert n_samples<=5000, 'Maximal 5000 simulations possible.'
        # Set up attributes
        self.time = time
        self.n_samples = n_samples
        # Get chunks
        chunks = [chunk_size for _ in range(n_samples//chunk_size)] 
        if n_samples%chunk_size > 0: 
            chunks += [n_samples%chunk_size]
        a = [(n, chunk) for n, chunk in enumerate(chunks)]
        # Integrate chunkwise in parallel
        with Pool(3) as p: 
            result = p.starmap(self._integrate, [(n+1, chunk) for n, chunk in enumerate(chunks)])
        hits_rt, fas_rt, trajectory, fa_trials = zip(*result)
        # Stack chunked results
        hits_rt = np.sum(np.stack(hits_rt), axis=0)
        fas_rt = np.sum(np.stack(fas_rt), axis=0)
        trajectory = np.vstack(trajectory)
        fa_trials = np.vstack(fa_trials)
        # Set as attribute
        self.hits_rt = hits_rt
        self.fas_rt = fas_rt
        self.trajectory = trajectory
        self.fa_trials = fa_trials
        return hits_rt, fas_rt, trajectory, fa_trials

    def _integrate(self, n, chunk_size):
        # Set up states, trajectory and rt array
        states = np.repeat(self.bias,chunk_size)
        trajectory = np.zeros((chunk_size, self.time))
        trajectory[:,0] = states
        hits_rt, fas_rt =  [np.zeros(self.time) for _ in range(2)]
        # Set up indexing
        hits, fas, finished = [np.zeros(chunk_size, dtype='bool') for _ in range(3)]
        # Load and adjust noise values
        noise = colored_noise(self.noise_color, low_cut=5, high_cut=495, number=n, time=self.time, n_samples=chunk_size)
        noise.trace *= self.noise_level
        for t in np.arange(1,self.time):
            # update states
            states[finished==False] += self.drift + noise[finished==False,t]
            # get reaction time
            new_hits = np.sum([states>=self.bound]) - np.sum(hits)
            hits_rt[t] = new_hits
            new_fas = np.sum([states<=0]) - np.sum(fas)
            fas_rt[t] = new_fas
            # update hits, fas and finished
            hits[states>=self.bound]=True; fas[states<=0]=True
            finished = (hits+fas).astype('bool')
            # save states to trajectory
            states[states>self.bound] = self.bound; states[states<0] = 0
            trajectory[:,t] = states
            
            # Return if all states are finished
            if np.all(finished):
                # Pad hit trajectories with ones
                pad_trajectory = np.zeros((chunk_size, self.time))
                pad_trajectory[hits, :] = 1
                pad_trajectory[:,:t+1] = trajectory[:,:t+1]
                return hits_rt, fas_rt, pad_trajectory.astype('float32'), fas
        
        return hits_rt, fas_rt, trajectory.astype('float32'), fas
 
    def plot_trajectory(self, n_trials, ax):
        # Set up boundaries
        for bound in 0, self.bound:
            ax.plot(np.repeat([bound], self.time), linewidth=0.5, color ='k')
        ax.plot(np.repeat([self.bias], self.time), 'k', linestyle=(0,(1,10)))
        # plot trajectories
        for trial in random.sample(self.trajectory.tolist(), n_trials):
            trial = np.array(trial)
            idx = np.diff(trial,prepend=1)!=0
            sns.lineplot(x = np.arange(0,len(trial[idx])), y=trial[idx],ax=ax, linewidth=0.2)
        # Configure ax
        ax.set_ylim(-0.1, self.bound+0.1)
        ax.set_yticks([0,self.bias, self.bound])
        ax.set_xlabel('Time [ms]')
        textstr = f'Example Trials = {n_trials}\nDrift = {self.drift}\nNoise Color = {self.noise_color.capitalize()}\nNoise STD = {self.noise_level}'
        ax.text(0.8, 0.85, textstr, transform=ax.transAxes, fontsize=15, linespacing=1.5,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='darkgray', alpha=0.8))
