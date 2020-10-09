import numpy as np
import matplotlib.pyplot as plt
def ddm(vel, thres, bias, n_level, time, axes):
    trajectory = [bias]
    for sec in np.arange(0,time):
        drift = vel + np.random.randn()*n_level
        val = trajectory[-1] + drift
        if val >= thres or val <= 0: 
            ax.plot(trajectory)            
            return {'result': int(val>=thres)*1, 'rt': sec,'trace': trajectory}
        else:
            trajectory.append(val)
    ax.plot(trajectory)
    return {'result': int(val>=thres)*1, 'rt': sec,'trace': trajectory}
    
fig, ax = plt.subplots()
ax.plot(np.repeat([1], 50), '--k')
for i in range(100):
    print(ddm(0.02, 1, 1/2, 0.025, 50, ax))
plt.show() 



