import numpy as np
class colored_noise(): 
    """
    Class to handle noise which is added to model
    """
    def __init__(self, params): 
        self.params = params
        noise_dict = {1:'white', 2:'pink', 3:'blue'}
        noise_color = noise_dict[params['nr_color']]
        noise_lowcut = params['noise_lowcut']
        noise_highcut = params['noise_highcut']
        self.csvfile = 'colored_noise/%s_noise_%d-%d.csv' % (noise_color, noise_lowcut, noise_highcut)
        self.noise_trace = self.load() 
    
    def __getattribute__(self, index):
        return self.noise_trace[index]

    def load(self):
        noise = np.genfromtxt(self.csvfile, delimiter=',')
        return noise

    def generate(self):
        #TODO translate matlab generator into python
        pass
