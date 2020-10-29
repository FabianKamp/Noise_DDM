import numpy as np
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from noise.noise_handling import colored_noise

class noise_ddm(object):
    def __init__(self, drift=0.05, bound=1, bias=0.5, noise_level=0.1, noise_color='white'):
        """
        Set parameters 
        """
        self.drift = drift
        self.bound = bound
        self.bias = bias
        self.noise_level = noise_level
        self.noise = colored_noise(noise_color, lowcut=5, highcut=495)
        self.trajectory = np.array([])
        self.rt = np.array([])
        self.n_samples = 0
        self.time = 60
    
    def get_noise(self, n_samples): 
        if self.noise_color == 'white': 
            return np.random.randn(n_samples)
        pass

    def simulate(self, time=60, n_samples=1000):
        for param in [self.drift, self.bound, self.bias, self.noise_level, self.noise_color]:
            assert not isinstance(param, list), 'Parameter range must be fitted before starting simulation.'
        # Set up attributes
        self.time=time
        self.n_samples=n_samples
        self.rt = np.zeros(time)
        self.trajectory = np.zeros((n_samples,time))
        
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
            trace[out==False] += self.drift + self.noise(:n_samples,t)*self.noise_level
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
        for t in self.trajectory[:n_trials]:
            idx = np.diff(t,prepend=1)!=0
            sns.lineplot(x = np.arange(0,len(t[idx])), y=t[idx],ax=ax)
        # Configure ax
        ax.set_ylim(-0.1, self.bound+0.1)
        ax.set_yticks([0,self.bias, self.bound])
        ax.set_xlabel('Time')
        ax.text(0.8, 0.85, f'# Trials = {n_trials}', transform=ax.transAxes, fontsize=15,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='darkgray', alpha=0.25))
        
    def plot_rt(self, ax):
        # Plot reaction times as bar plot
        r = self.rt.astype(int); 
        t = np.arange(0,self.time)
        h = np.repeat(np.arange(0, self.time), r)
        sns.histplot(h, binwidth=1, ax=ax, alpha=0.5, kde=True, line_kws={'alpha':0.75})
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, nr: val/self.n_samples))
        ax.set_ylabel('Probability')
        ax.text(0.8, 0.85, f'# Samples = {self.n_samples}', transform=ax.transAxes, fontsize=15,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='darkgray', alpha=0.25))
        #sns.barplot(x = np.arange(0, self.time), y = self.rt, ax=ax, alpha=0.75, color='tab:blue')
    
    def plot_combined(self, n_trials):
        sns.set_style('white'); sns.set_palette('deep')
        fig = plt.figure()
        gs = fig.add_gridspec(3,1, hspace=0.0)
        ax = []
        ax.append(fig.add_subplot(gs[0,0]))
        ax.append(fig.add_subplot(gs[1:,0]))
        self.plot_rt(ax[0])
        self.plot_trajectory(n_trials, ax[1])
        ax[0].set_xticks([])
        return fig

if __name__ == "__main__":
    m = noise_ddm(bias=0.5)
    m.simulate(n_samples=1000, time=10000)
    m.plot_combined(n_trials=50)
    plt.show()




