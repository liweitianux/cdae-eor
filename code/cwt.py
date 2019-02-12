
# coding: utf-8

# # EoR Signal Separation with CWT Method
# 
# ---
# 
# ### Weitian LI
# 
# https://github.com/liweitianux/cdae-eor
# 
# **Credit**:
# [Junhua GU](https://github.com/astrojhgu)

# ---
# 
# ## Introduction
# 
# The foreground spectra are smooth in frequency domain, while the EoR signal fluctuates rapidly along
# the frequency dimension, i.e., its spectrum is full of saw-tooth-like structures.  Therefore their
# characteriestic scales are significantly different.  By applying the continuous wavelet transform (CWT),
# they should be well separated.
# 
# **Reference**:
# [Gu et al. 2013, ApJ, 773, 38](http://adsabs.harvard.edu/abs/2013ApJ...773...38G)

# ---
# 
# ## 1. Import packages and basic settings

# In[1]:


import os
from os import path

import numpy as np
from scipy import signal
from astropy.io import fits


# In[2]:


import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


mpl.style.use("ggplot")
for k, v in [("font.family",       "Inconsolata"),
             ("font.size",         14.0),
             ("pdf.fonttype",      42),  # Type 42 (a.k.a. TrueType)
             ("figure.figsize",    [8, 6]),
             ("image.cmap",        "jet"),
             ("xtick.labelsize",   "large"),
             ("xtick.major.size",  7.0),
             ("xtick.major.width", 2.0),
             ("xtick.minor.size",  4.0),
             ("xtick.minor.width", 1.5),
             ("ytick.labelsize",   "large"),
             ("ytick.major.size",  7.0),
             ("ytick.major.width", 2.0),
             ("ytick.minor.size",  4.0),
             ("ytick.minor.width", 1.5)]:
    mpl.rcParams[k] = v


# In[4]:


import sys

p = path.expanduser('~/git/cdae-eor/cwt/pycwt1d')
if p not in sys.path:
    sys.path.insert(0, p)

import cwt1d


# ---
# 
# ## 2. Custom functions

# In[5]:


def rms(a, axis=None):
    return np.sqrt(np.mean(a**2, axis=axis))


# In[6]:


def a_summary(a):
    print('min:', np.min(a))
    print('max:', np.max(a))
    print('mean:', np.mean(a))
    print('std:', np.std(a))
    print('median:', np.median(a))


# In[7]:


# correlation coefficient

def corrcoef(s1, s2):
    # calculate: np.corrcoef(s1, s2)[0, 1]
    m1 = np.mean(s1)
    m2 = np.mean(s2)
    return np.sum((s1-m1) * (s2-m2)) / np.sqrt(np.sum((s1-m1)**2) * np.sum((s2-m2)**2))


def corrcoef_ds(ds1, ds2):
    # shape: [npix, nfreq]
    n = ds1.shape[0]
    cc = np.zeros((n,))
    for i in range(n):
        cc[i] = corrcoef(ds1[i, :], ds2[i, :])
    return cc


def corrcoef_freqpix(fparray1, fparray2):
    # shape: [nfreq, npix]
    __, npix = fparray1.shape
    cc = np.zeros((npix,))
    for i in range(npix):
        cc[i] = corrcoef(fparray1[:, i], fparray2[:, i])
    return cc


# ---
# 
# ## 3. Load data

# In[8]:


datadir = '../data'
cube_eor = fits.open(path.join(datadir, 'eor.uvcut.sft_b158c80_n360-cube.fits'))[0].data.astype(float)
cube_fg  = fits.open(path.join(datadir, 'fg.uvcut.sft_b158c80_n360-cube.fits' ))[0].data.astype(float)
cube_tot = cube_fg + cube_eor


# In[9]:


nfreq, ny, nx = cube_eor.shape
npix = nx * ny
freqs = np.linspace(154, 162, nfreq)
fmid = (freqs[1:] + freqs[:-1]) / 2

nfreq, ny, nx, npix


# In[10]:


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

ax = ax0
eor_rms = rms(cube_eor, axis=(1,2)) * 1e3  # mK
ax.plot(freqs, eor_rms, lw=2.5, label='rms')
ax.legend()
ax.set(xlabel='Frequency [MHz]', ylabel='Tb [mK]', title='EoR')

ax = ax1
fg_rms = rms(cube_fg, axis=(1,2))
ax.plot(freqs, fg_rms, lw=2.5, label='rms')
ax.legend()
ax.set(xlabel='Frequency [MHz]', ylabel='Tb [K]', title='Foreground')
ax_ = ax.twinx()
ax_.plot(fmid, np.diff(fg_rms)*1e3, color='C1', label='diff')
ax_.legend()
ax_.set(ylabel='diff(Tb) [mK]')
ax_.grid(False)

fig.tight_layout()
plt.show()


# ---
# 
# ## 4. Tune parameters

# In[11]:


x_input = np.array(cube_tot.reshape((nfreq, npix)))
x_label = np.array(cube_eor.reshape((nfreq, npix)))


# In[12]:


x1 = x_input[:, 0]
y1 = x_label[:, 0]


# In[13]:


fig, ax = plt.subplots()
ax.plot(freqs, x1, color='C0', label='FG+EoR')
ax.legend()
ax_ = ax.twinx()
ax_.plot(freqs, y1, color='C1', label='EoR')
ax_.legend()
plt.show()


# In[60]:


def plot_cwt(mask, coef):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(14, 4))
    ax0.imshow(mask, origin='lower')
    ax0.grid(False)
    ax0.set(title='mask')
    ax1.imshow(np.abs(coef), origin='lower')
    ax1.grid(False)
    ax1.set(title='coefficient')
    fig.tight_layout()
    plt.show()
    return (fig, (ax0, ax1))


def test_cwt(data, coi, s_min, s_max, num_scales=50, nig=10, plot=True):
    xin, xlabel = data
    nfreq = len(xin)
    mwf = cwt1d.morlet(2*np.pi)
    scale = cwt1d.generate_log_scales(s_min, s_max, num_scales)
    mask = cwt1d.cwt_filter.generate_mask(nfreq, scale, coi)
    coef = cwt1d.cwt(x1, scale, mwf)
    xout = cwt1d.icwt(coef*mask, scale, mwf)
    
    if plot:
        plot_cwt(mask, coef)
    cc = corrcoef(xout[nig:-nig], xlabel[nig:-nig])
    print(f'cc: {cc:.4f}')
    
    return {
        'xout': xout,
        'mask': mask,
        'coef': coef,
        'cc': cc,
    }


# In[48]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=3, s_max=50)


# In[49]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=3, s_max=50, num_scales=100)


# In[51]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=3, s_max=50, num_scales=30)


# In[50]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=1, s_max=50)


# In[52]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=10, s_max=50)


# In[54]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=3.8, s_max=50)


# In[55]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=3.8, s_max=30)


# In[56]:


ret = test_cwt(data=(x1, y1), coi=3, s_min=1, s_max=50)


# In[58]:


ret = test_cwt(data=(x1, y1), coi=1, s_min=1, s_max=50)


# In[59]:


ret = test_cwt(data=(x1, y1), coi=5, s_min=1, s_max=50)


# In[63]:


for p in np.arange(1, 5, 0.1):
    print(f'coi={p:.1f} ... ', end='', flush=True)
    ret = test_cwt(data=(x1, y1), coi=p, s_min=1, s_max=50, plot=False)


# In[64]:


coi = 1.6
for p in np.arange(1, 10, 0.2):
    print(f's_min={p:.1f} ... ', end='', flush=True)
    ret = test_cwt(data=(x1, y1), coi=coi, s_min=p, s_max=50, plot=False)


# In[68]:


coi = 1.6
s_min = 7.4
for p in np.arange(30, 100, 2.0, dtype=float):
    print(f's_max={p:.1f} ... ', end='', flush=True)
    ret = test_cwt(data=(x1, y1), coi=coi, s_min=s_min, s_max=p, plot=False)


# In[73]:


coi = 1.6
s_min = 7.4
s_max = 50.0
for p in np.arange(30, 100, 2, dtype=np.int32):
    print(f'num_scales={p} ... ', end='', flush=True)
    ret = test_cwt(data=(x1, y1), coi=coi, s_min=s_min, s_max=s_max, num_scales=p, plot=False)


# In[76]:


coi = 1.6
s_min = 7.4
s_max = 50.0
num_scales = 50

ret = test_cwt(data=(x1, y1), coi=coi, s_min=s_min, s_max=s_max, num_scales=num_scales)


# In[79]:


fig, ax = plt.subplots()
ax.plot(freqs, y1, lw=2, label='input')
ax.plot(freqs, ret['xout'], lw=2, label='output')
ax.legend()
plt.show()


# ---
# 
# ## 5. Results

# In[97]:


nig = 10
cwt_args = {
    'coi': coi,
    's_min': s_min,
    's_max': s_max,
    'num_scales': num_scales,
    'nig': nig,
}


# In[89]:


def fgrm_cwt(x_input, **kwargs):
    if x_input.ndim == 1:
        nfreq = len(x_input)
        npix = 1
    else:
        nfreq, npix = x_input.shape
    mwf = cwt1d.morlet(2*np.pi)
    scale = cwt1d.generate_log_scales(kwargs['s_min'], kwargs['s_max'], kwargs['num_scales'])
    mask = cwt1d.cwt_filter.generate_mask(nfreq, scale, kwargs['coi'])
    
    if npix == 1:
        coef = cwt1d.cwt(x_input, scale, mwf)
        return cwt1d.icwt(coef*mask, scale, mwf)
    
    out = np.zeros((nfreq, npix))
    percent = npix // 100
    for i in range(npix):
        if npix > 1e3 and i % percent == 0:
            print('%d..' % (i//percent), end='', flush=True)
        coef = cwt1d.cwt(x_input[:, i], scale, mwf)
        out[:, i] = cwt1d.icwt(coef*mask, scale, mwf)
    if npix > 1e3:
        print('', flush=True)
    
    return out


# #### 5% dataset

# In[85]:


idx = np.arange(npix)
np.random.seed(42)
np.random.shuffle(idx)

n = int(npix * 0.05)  # 5%
x_idx = idx[:n]
x_tot = x_input[:, x_idx]
x_eor = x_label[:, x_idx]
x_eor.shape


# In[88]:


get_ipython().run_cell_magic('time', '', 'x_out = fgrm_cwt(x_tot, **cwt_args)')


# In[100]:


cc = corrcoef_freqpix(x_out[nig:-nig, :], x_eor[nig:-nig, :])
print('rho: %.4f +/- %.4f' % (cc.mean(), cc.std()))
np.mean(np.abs(cc)), np.std(cc), rms(cc), np.percentile(cc, q=(25, 50, 75))


# #### 20% dataset

# In[101]:


idx = np.arange(npix)
np.random.seed(42)
np.random.shuffle(idx)

n = int(npix * 0.2)  # 20%
x_idx = idx[:n]
x_tot = x_input[:, x_idx]
x_eor = x_label[:, x_idx]
x_eor.shape


# In[102]:


get_ipython().run_cell_magic('time', '', 'x_out = fgrm_cwt(x_tot, **cwt_args)')


# In[103]:


cc = corrcoef_freqpix(x_out[nig:-nig, :], x_eor[nig:-nig, :])
print('rho: %.4f +/- %.4f' % (cc.mean(), cc.std()))
np.mean(np.abs(cc)), np.std(cc), rms(cc), np.percentile(cc, q=(25, 50, 75))


# #### full dataset

# In[104]:


get_ipython().run_cell_magic('time', '', 'x_out = fgrm_cwt(x_input, **cwt_args)')


# In[105]:


cc = corrcoef_freqpix(x_out[nig:-nig, :], x_label[nig:-nig, :])
print('rho: %.4f +/- %.4f' % (cc.mean(), cc.std()))
np.mean(np.abs(cc)), np.std(cc), rms(cc), np.percentile(cc, q=(25, 50, 75))

