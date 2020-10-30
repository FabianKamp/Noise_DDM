import numpy as np
import scipy
import os

class colored_noise(): 
	"""
	Class to handle noise which is added to model
	"""
	def __init__(self, noise_color, low_cut, high_cut, time, n_samples): 
		self.noise_color = noise_color
		self.low_cut = low_cut
		self.hight_cut = high_cut
		self.fsample = 1000
		self.time = time
		self.n_samples = n_samples
		self.csvfile = os.path.join('noise',f'{noise_color}_noise_{low_cut}-{high_cut}.csv')
		self.noise_trace = self.load()
		self.noise_trace = self.noise_trace[:n_samples, :time]
	
	def __getitem__(self, index):
		return self.noise_trace[index]

	def load(self):
		noise = np.genfromtxt(self.csvfile, delimiter=',')
		return noise

	def generate(self):
		#TODO translate matlab generator into python
		pass
	
	def plot_noise(self):
		pass
	
	def plotNoiseSpectrum(self, ax):
		"""
		Plots a Single-Sided Amplitude Spectrum of Noise with Sampling Frequency Fsample
		"""
		n = len(self.noise_trace) # length of the Noise
		k = np.arange(n)
		T = n/self.fsample
		frq = k/T # two sides frequency range
		frq = frq[range(int(n/2))] # one side frequency range

		Y = scipy.fft(Noise) # fft computing and normalization
		Y = Y[range(int(n/2))]
		
		ax.plot(frq,np.abs(Y), alpha = 0.50) # plotting the spectrum		
		ax.set_xlabel('Frequency (Hz)')
		#ax.tick_params(labelbottom=False)

		ax.set_ylabel('Power', fontsize='small')
		ax.yaxis.set_label_position("right")
		ax.yaxis.tick_right()

		
