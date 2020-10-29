import numpy as np
import matplotlib.pyplot as plt
from ddm import Model, Fittable
from ddm.models import LossRobustBIC, DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision
from ddm.functions import fit_adjust_model, display_model
import ddm.plot

m = Model(name='Simple', drift=DriftConstant(drift=2.2), 
            noise=NoiseConstant(noise=1.5), 
            bound=BoundConstant(B=1.1),
            overlay=OverlayNonDecision(nondectime=.1), 
            dx=.001, dt=.01, T_dur=2)
display_model(m)
sol = m.solve()
samp = sol.resample(1000)

f = Model(name='Fit', drift=DriftConstant(drift=Fittable(minval=0, maxval=4)), 
            noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)), 
            bound=BoundConstant(B=1.1),
            overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)), 
            dx=.001, dt=.01, T_dur=2)

fit_adjust_model(samp, f, fitting_method="differential_evolution", 
                    lossfunction=LossRobustBIC, verbose=False)

display_model(f)

ddm.plot.plot_fit_diagnostics(f, samp)
plt.show()