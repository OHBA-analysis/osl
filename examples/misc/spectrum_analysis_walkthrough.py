import osl_ephys
from scipy import signal
import matplotlib.pyplot as plt

raw = osl_ephys.utils.simulate_raw_from_template(10000, noise=1/3)
raw.pick(picks='mag')


#%%
spec = osl_ephys.glm.glm_spectrum(raw)
spec.plot_joint_spectrum(freqs=(1, 10, 17), base=0.5, title='testing123')

#%%
alpha = raw.copy().filter(l_freq=7, h_freq=13)
covs = {'alpha': np.abs(signal.hilbert(alpha.get_data()[raw.ch_names.index('MEG1711'), :]))}

spec = osl_ephys.glm.glm_spectrum(raw, reg_ztrans=covs)

plt.figure()
ax = plt.subplot(121)
spec.plot_joint_spectrum(0, freqs=(1, 10, 17), base=0.5,ax=ax)
ax = plt.subplot(122)
spec.plot_joint_spectrum(1, freqs=(1, 10, 17), base=0.5,ax=ax)




gglmsp = osl_ephys.glm.read_glm_spectrum('/Users/andrew/Downloads/bigmeg-camcan-movecomptrans_glm-spectrum_grad-noztrans_group-level.pkl')
spec = osl_ephys.glm.GroupSensorGLMSpectrum(gglmsp.model, 
                                      gglmsp.design, 
                                      gglmsp.config, 
                                      gglmsp.info, 
                                      fl_contrast_names=None, 
                                      data=gglmsp.data)
P = osl_ephys.glm.MaxStatPermuteGLMSpectrum(spec, 1, nperms=25)
