# absolutely necessary for core functionalities
import numpy as np
import numpy.random as rn
import numpy.ma as ma
import pywt
# from sklearn import preprocessing as pp
# from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import shift
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import callbacks, initializers, layers, metrics, models, optimizers
from tensorflow.python.keras import backend as K

# necessary for all code to run, could be removed if not used
import pickle
import glob     # for listing files to open
import re       # for using regular expressions in finding files
from astropy.io import fits                                             # for opening fits files
from scipy.signal import savgol_filter, welch, periodogram    # for showing smoothed data and creating mock data with lines
import matplotlib.pyplot as plt                                         # for plotting
from matplotlib.ticker import ScalarFormatter

import time                     # for showing the time spent on training
import ipywidgets as wdg        # for interactive plots

# optional
plt.style.use('seaborn')        # plot style
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))       # wider notebook
import warnings                 # no large red boxes with warnings in the notebook
warnings.filterwarnings('ignore')

class Split(layers.Layer):
    """Custom split layer to split layer into two equally (or almost equally) sized layers."""
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def build(self, input_shape):
        # Call the build function of the parent class to build this layer
        super(Split, self).build(input_shape)
        # Save the shape, use other functions
        self.shape = input_shape

    def call(self, x, mask=None):
        # Split x into two tensors
        seq = [x[:, 0:self.shape[1] // 2, ...],
               x[:, self.shape[1] // 2:, ...]]
        return seq

    def compute_mask(self, inputs, input_mask=None):
        # This layer outputs two tensors and needs to return multiple masks, and the mask can be None
        return [None, None]

    def get_output_shape_for(self, input_shape):
        # If this layer returns two tensors, it will return the shape of the two tensors
        shape0 = list(self.shape)
        shape1 = list(self.shape)
        shape0[1] = self.shape[1] // 2
        shape1[1] = self.shape[1] - self.shape[1] // 2
        # print [shape0, shape1]
        return [shape0, shape1]


class Data:
    def __init__(self, freq=None, flux=None, hdul=None, subbands=None, sub_len=None, file_len=None, X=None):
        """
        Holds the frequency data before and after using the neural network.
        freq:       frequencies
        flux:       fluxes
        hdul:       Header Data Unit list of .fits files
        subbands:   which subbands are used (for HIFI it's a selection of 1 through 4)
        sub_len:    respective length of each subband (number of channels)
        X:          merged, scaled data; this should be the output of the neural network
        """
        self.freq = freq
        self.flux = flux
        self.hdul = hdul
        self.subbands = subbands
        self.sub_len = sub_len
        self.file_len = file_len
        if hdul is not None:
            self.cols = hdul[1].columns
            self.u_freq = self.cols['frequency_2'].unit 
            self.u_flux = self.cols['flux_2'].unit
        
        self.X = X
        
        self.flux_c = None
        self.freq_c = None
        self.X_unscaled = None
        
        
    def merge(self, cutoff=True, freqpad=0, freqlims=False):
        """
        Returns a numpy array with the frequency data from all subbands together. Thus it has one spectrum per observation.
        As there is overlap between the subbands, the spectra have to either be merged or concatenated.
        
        If cutoff is True, some overlapping frequencies will be discarded.
        
        If cutoff is False, the spectra will be concatenated. The overlapping frequencies will be repeated.
        
        freqpad cuts off the outer parts of the spectrum, which is recommended if the data is used for training as this will allow new data to fully overlap
        alternatively, freqlims gives the frequency limits directly, which is used for observational data
        """
#         def match_shape(arr):
#             """Turns list of differently shaped arrays into one array filled with nan values."""
# #             arr = np.array(arr)
# #             if np.ndim(arr) == 1:
#             max_shape = np.max([np.shape(a) for a in arr], axis=0)
#             new_arr = np.zeros((0, *max_shape))
#             for a in arr:
#                 a_new = np.zeros(max_shape)
#                 a_new[:] = np.nan
#                 s = np.shape(a)
#                 a_new[[slice(s_i) for s_i in s]] = a
#                 new_arr = np.append(new_arr, a_new[np.newaxis, ...], axis=0)
#             return new_arr
# #         else:
# #             return arr
        
#         freq = match_shape(self.freq)
#         flux = match_shape(self.flux)
        
#         # turn flux and freq into masked array, where inf is used as filler (to get the right shape), and NaN for invalid data
#         self.flux = ma.masked_array(flux, mask=((flux==np.infty) + np.isnan(flux)))
#         self.freq = ma.masked_array(freq, mask=((freq==np.infty) + np.isnan(freq) + (freq==0)))
#         if cutoff:
#             # frequency range of each subband
#             freq_ranges = np.array([np.nanmin(self.freq, axis=4)[0,0,:,0], np.nanmax(self.freq, axis=4)[0,0,:,0]])
            
#             # frequencies in the middle of an overlapping part of the spectrum
#             intersections = [(freq_ranges[0, i+1] + freq_ranges[1, i])/2 for i in range(len(freq_ranges))]  
#             print('frequency ranges of subbands:', freq_ranges)
#             print('subband intersections:', [float('{:6.1f}'.format(i)) for i in intersections], self.u_freq)
            
#             # boolean to select frequencies that do not cross the intersection
#             no_overlap = (self.freq[0][0, :, 0, :] > np.array([0, *intersections])[:, np.newaxis]) & \
#                          (self.freq[0][0, :, 0, :] < np.array([*intersections, np.infty])[:, np.newaxis])
            
#             # apply boolean to frequency and flux data to get merged spectra
#             self.freq_c = self.freq.swapaxes(2,3)[:, :, :, no_overlap]     # use frequencies of 1 spectrum; they're all the same
#             self.flux_c = self.flux.swapaxes(2,3)[:, :, :, no_overlap]
            
#             # simplify shape
#             s = np.shape(self.flux_c)
#             self.freq_c = np.reshape(self.freq_c, (np.product(s[:-1]), -1))
#             self.flux_c = np.reshape(self.flux_c, (np.product(s[:-1]), -1))
#             print('data shape simplified')
            
            
#             # cut off edges of spectra and regrid to fixed step size frequency grid to get overlapping frequencies
#             freqlims_files = np.asarray([(np.min(f), np.max(f)) for f in self.freq_c])     # boundaries of frequencies for each file
#             self.freqlims = np.nanmax(freqlims_files[:, 0] + freqpad), np.nanmin(freqlims_files[:, 1] - freqpad)  # new boundaries for all files
#             if type(freqlims) is not bool:      # use freqlims if given
#                 self.freqlims = freqlims
            
#             freqgrid = np.arange(self.freqlims[0], self.freqlims[1] + 0.5, 0.5)
#             X = []
#             for x, y in zip(self.freq_c, self.flux_c):
#                 try:
#                     f = interp1d(x, y, kind='linear', axis=-1)
#                     X.append(f(freqgrid))
#                 except: 
#                     pass
#             X = np.asarray(X)
#             self.freq_c = freqgrid
#             self.X = X
#             print('regridded and restricted frequency range')
            
# #             mask_freqlims = (self.freq_c.data <= self.freqlims[0]) + (self.freq_c.data >= self.freqlims[1])
            
# #             mask_other = self.flux_c.mask              # for NaN values
# #             self.mask_freqlims = mask_freqlims

            
# #             s = np.shape(self.freq_c)

# #             self.freq_c = self.freq_c.data[~mask_freqlims]
# #             self.flux_c = self.flux_c.data[~mask_freqlims]
         
# #             # simplify shape again
# #             self.freq_c = np.reshape(self.freq_c, (s[0], -1))
# #             self.flux_c = np.reshape(self.flux_c, (s[0], -1))
# #             mask_other = np.reshape(mask_other, (s[0], -1))
# #             self.freq_c.mask = mask_other
# #             self.flux_c.mask = mask_other
            
    
    
    
#             # reshape to two dimensions and get rid of filler values (infinite values)
# #             X = np.reshape(self.flux_c.data[self.flux_c.data!=np.infty], (-1, s[-1]))
    
#             # get rid of NaN values
# #             self.X = X.data[np.any(np.isnan(X.data), axis=1)]
    
#         if not cutoff:
#             flux_c = []
#             for f in self.flux:
#                 f = f.swapaxes(0, 1)        # set the subband axis to be the first axis
#                 f_sub_list = []
#                 for f_sub, l in zip(f, self.sub_len):            # loop the data over the subband axis, and the length l of each subband
#                     f_sub = f_sub[..., :l]                      # shorten the data by only using the data within the subband length 
#                     f_sub_list.append(f_sub)                    # add the shortened data to a list

#                 f = np.concatenate(f_sub_list, axis=-1)     # merge the data together
#                 flux_c.append(f)
#             self.flux_c = flux_c
#             self.X = self.flux_c
              
                
        def match_shape(arr):
            """Turns list of differently shaped arrays into one array filled with nan values."""
#             arr = np.array(arr)
#             if np.ndim(arr) == 1:
            max_shape = np.max([np.shape(a) for a in arr], axis=0)
            new_arr = np.zeros((0, *max_shape))
            for a in arr:
                a_new = np.zeros(max_shape)
                a_new[:] = np.nan
                s = np.shape(a)
                a_new[[slice(s_i) for s_i in s]] = a
                new_arr = np.append(new_arr, a_new[np.newaxis, ...], axis=0)
            return new_arr
        
        freq = match_shape(self.freq)
        flux = match_shape(self.flux)
        
        # turn flux and freq into masked array, where inf is used as filler (to get the right shape), and NaN for invalid data
        self.flux = ma.masked_array(flux, mask=((flux==np.infty) + np.isnan(flux)))
        self.freq = ma.masked_array(freq, mask=((freq==np.infty) + np.isnan(freq) + (freq==0)))
        if cutoff:
            # simplify shape
            fr = self.freq.swapaxes(-3, -2)
            fl = self.flux.swapaxes(-3, -2)
            s = np.shape(fr)
            fr = np.reshape(fr, (-1, len(self.subbands), s[-1]))
            fl = np.reshape(fl, (-1, len(self.subbands), s[-1]))
            
            # merge subbands
            for i, _ in enumerate(fr):
                # find frequency ranges and subband intersections for each spectrum
                freqranges = np.asarray([np.nanmin(fr[i], axis=1), np.nanmax(fr[i], axis=1)])
                intersections = [(freqranges[0, i+1] + freqranges[1, i])/2 for i in range(len(freqranges))] 

                # mask values outside of intersection bounds
                for k, _ in enumerate(fr[i]):
                    if k !=0:
                        try:
                            fr[i].mask[k] += (subfreq < intersections[k-1])
                            fl[i].mask[k] += (subfreq < intersections[k-1])
                        except: pass
                    if k != -1:
                        try:
                            fr[i].mask[k] += (subfreq > intersections[k])
                            fl[i].mask[k] += (subfreq > intersections[k])
                        except: pass
                    
            # get rid of subband dimension
            s = np.shape(fr)
            self.freq_c = fr.reshape(s[0], -1)
            self.flux_c = fl.reshape(s[0], -1)
            print('data shape simplified')
            
            
            # cut off edges of spectra and regrid to fixed step size frequency grid to get overlapping frequencies
            freqlims_files = np.asarray([(np.min(f), np.max(f)) for f in self.freq_c])     # boundaries of frequencies for each file
            self.freqlims = np.nanmax(freqlims_files[:, 0] + freqpad), np.nanmin(freqlims_files[:, 1] - freqpad)  # new boundaries for all files
            if type(freqlims) is not bool:      # use freqlims if given
                self.freqlims = freqlims
            
            freqgrid = np.arange(self.freqlims[0], self.freqlims[1] + 0.5, 0.5)
            X = []
            for x, y in zip(self.freq_c, self.flux_c):
                try:
                    f = interp1d(x, y, kind='linear', axis=-1)
                    X.append(f(freqgrid))
                except: 
                    pass
            X = np.asarray(X)
            self.freq_c = freqgrid
            self.X = X
            print('regridded and restricted frequency range')
    
        if not cutoff:
            flux_c = []
            for f in self.flux:
                f = f.swapaxes(0, 1)        # set the subband axis to be the first axis
                f_sub_list = []
                for f_sub, l in zip(f, self.sub_len):            # loop the data over the subband axis, and the length l of each subband
                    f_sub = f_sub[..., :l]                      # shorten the data by only using the data within the subband length 
                    f_sub_list.append(f_sub)                    # add the shortened data to a list

                f = np.concatenate(f_sub_list, axis=-1)     # merge the data together
                flux_c.append(f)
            self.flux_c = flux_c
            self.X = self.flux_c
            
        
    def scale(self, scale_type='minmax_zero', individual=False):
        X = self.X
        if individual:      # scale each fits file seperately
#             h = 0           # l, h are the lowest and highest index of the spectra from one fits file
#             scale_params = []
#             for fl in self.file_len:
#                 l = h
#                 h = l + fl
#                 X[l:h], s = scale(X[l:h], scale_type)
#                 scale_params.append(s)
            scale_params = []
            for i, x in enumerate(X):
                X[i], s = scale(x, scale_type)
                scale_params.append(s)
        else:
            X, scale_params = scale(self.X, scale_type)
        
        self.X = X
        print('data scaled using {}'.format(scale_type))
        return scale_params
    
    
    def rescale(self, scale_params, scale_type='normal'):
        self.X_unscaled = rescale(self.X, scale_params, scale_type)
        
        
    def remove_empty(self, fillnan=True, delta=0.02):
        """Discard spectra with same values everywhere. If fillnan is true, channels with NaN values are interpolated.
        The difference between the minimum and maximum value should be larger than delta."""
        X = self.X
        
        if fillnan:
            mask = np.isnan(X)
            X[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), X[~mask])
        
        # remove empty spectra
        same_mask = [np.nanmax(X, axis=1) - np.nanmin(X, axis=1) < delta]
        same_mask = np.tile(same_mask, (X.shape[-1], 1)).T       
        X = ma.masked_array(X, mask=same_mask)
        self.X = X
        
        
    def plot_spectrum(self):
        """
        Interactive plot to see spectra, and to switch between files and observations.
        """
        sub_len = self.sub_len
        flux = self.flux
        freq = self.freq
        subbands = self.subbands
        
        # plot
        notebook()        
        plt.figure(figsize=(8, 3))
        lw = .5
        lines = []
        for i, sub in enumerate(subbands):
            line, = plt.plot(freq[0][0, i, 0, :sub_len[i]], flux[0][0, i, 0, :sub_len[i]], 
                             lw=lw, label='subband {}'.format(sub))
            lines.append(line)

        plt.xlim(5400, 8100)
        plt.ylim(-20, 20)
        leg = plt.legend(loc=1)
        for l in leg.get_lines():
            l.set_linewidth(2)
        plt.xlabel('frequency ({})'.format(self.u_freq))
        plt.ylabel('flux ({})'.format(self.u_flux))
        plt.title('Spectrogram')
        fig.tight_layout()

        # update function
        def update_plot(file, obs, obs2):
            for i, line in enumerate(lines):
                x = freq[file][obs, i, obs2]
                y = flux[file][obs, i, obs2]
                line.set_xdata(x[:sub_len[i]])
                line.set_ydata(y[:sub_len[i]])

        # widgets
        s = np.shape(flux)
        w0 = wdg.IntSlider(value=0, min=0, max=s[-5]-1, step=1)
        w1 = wdg.IntSlider(value=0, min=0, max=s[-4]-1, step=1)
        w2 = wdg.IntSlider(value=0, min=0, max=s[-2]-1, step=1)
        wdg.interact(update_plot, file=w0, obs=w1, obs2=w2)
        
    
    def plot_scans(self, file=0, obs=15):
        """Plot multiple scans of the same observation. obs is the observation number, file is the file number"""
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        plt.figure(figsize=(8, 4), dpi=120)
        for k in range(min(len(colors), np.shape(self.freq)[3])):
            for i in range(np.shape(self.freq)[2]):
                plt.plot(self.freq[file][obs, i, k], self.flux[file][obs, i, k], lw=.3, color=colors[k])

        plt.xlim(5400, 8100)
        plt.ylim(-15, 15)
        plt.xlabel('intermediate frequency ({})'.format(self.u_freq))
        plt.ylabel('flux ({})'.format(self.u_flux))
        plt.title('IF spectrogram for multiple scans')
        plt.show()
       
    
    def plot_fourier(self):
        """Interactive plot showing the FFT of input data."""
        # plot
        notebook()
        plt.figure()
        line, = plt.plot(np.fft.fft(self.X[0]), lw=.2)
        plt.xlim(0, self.X.shape[1])
        plt.ylim(-5, 5)
        plt.show()
        
        # update function
        def update_plot(i):
            line.set_ydata(np.fft.fft(self.X[i]))
        
        # widgets
        w0 = wdg.IntSlider(value=0, min=0, max=self.X.shape[0]-1, step=1)
        wdg.interact(update_plot, i=w0)
        
        
class NeuralNetwork:
    """
    Holds the model parameters from a neural network, and metadata such as history and the parameters used for scaling the data.
    """
    def __init__(self, model, history=None, scale_type='minmax_zero', scale_params=None, index=None, clip_range=None, freqlims=None):
        self.model = model
        self.scale_type = scale_type
        self.scale_params = scale_params            # scale parameters used to scale training data
        self.clip_range = clip_range                # clipped channel range
        self.freqlims = freqlims
        try:        # load history from model if available (doesn't work if the model is loaded using load_NN)
            self.history = model.history.history    
        except:     # load history directly from input parameters
            self.history = history
            
        
    def plot_test(self, X_test, smoothed=True):
        """
        Plot predictions against test data in an interactive plot. 
        By default, the plotted test data is smoothed using a Savitzky–Golay filter for visual clarity.
        """
        prediction = self.model.predict(X_test)     # prediction of test data using the neural network
        if smoothed:                                
            X_test = savgol_filter(X_test, 101, 3, axis=1)  # smoothed test data
        
        notebook()
        plt.figure(figsize=(5, 2), dpi=150)
        l1, = plt.plot(X_test[0], label='test data', lw=1)
        l2, = plt.plot(prediction[0], label='predicted', lw=1)
        lines = [l1, l2]
        plt.legend(loc=1)
        plt.ylim(np.min(X_test), np.max(X_test))
        plt.xlabel('channel')
        plt.ylabel('scaled flux')
        
        def update_plot(i):
            for k, dat in enumerate([X_test, prediction]):
                lines[k].set_ydata(dat[i])

        w0 = wdg.IntSlider(value=0, min=0, max=len(X_test)-1, step=1, layout=wdg.Layout(width='50%', height='20px'))
        wdg.interact(update_plot, i=w0)
        

    def plot_history(self, log=True, figsize=(8, 3), dpi=100):
        """Plots the metrics for each epoch, such as loss function and accuracy"""
        history = self.history
        keys = history.keys()
        titles = [k for k in keys if k[:3] != 'val']    # get metric names
        
        fig, ax = plt.subplots(1, len(titles), figsize=figsize, dpi=dpi)
        for i, t in enumerate(titles):
            a = ax[i] if type(ax) == np.ndarray else ax
            a.plot(history[t], label='training set')    # plot metric
            a.set_title(t)
            try:    # add metrics from validation set if given
                plot_data = history['val_{}'.format(t)]
                a.plot(plot_data, label='test set')
                if t == 'loss':
                    a.set_title(t + ' (min = {:.2e})'.format(np.min(plot_data)))
            except:
                pass
            a.set_xlabel('epoch')
            a.legend(loc=1)
            if t == 'cosine_proximity':
                a.set_ylim(bottom=-1)
            else:
                if log: a.set_yscale('log')
        fig.tight_layout()
        plt.show()
        

    def save(self, freqlims, filename1=None, filename2=None):
        """Saves the model to a .h5 or .hdf5 file, and its history to a .npy file."""
        if filename1 is not None:
#             try:
            self.model.save(filename1)
            print('model saved to {}'.format(filename1))
#             except:
#                 print('could not save model')
        if filename2 is not None:
            try:
                np.savez(filename2, self.history, self.scale_type, self.scale_params, freqlims)
                print('metadata saved to {}.npz'.format(filename2))
            except:
                print('could not save model metadata')

        
def load_NN(index=None, directory='nets', common='ann_', filetype='hdf5', fullpathname=None):
    """
    Load the model and history if given.
    The file structure should be [directory]/[common].h5 and .npy for the model and history respectively.
    """
    if fullpathname == None:        # use index if specified
        filename1 = '{}/{}{:04d}'.format(directory, common, index)
    else:                           # use full path name if specified
        filename1 = fullpathname

    print(filename1 + '.' + filetype)
    model = models.load_model(filename1 + '.' + filetype, 
                              custom_objects={'Split':Split})                  # load model from .hdf5 file
    print('model loaded from', filename1)
        
    filename2 = filename1 + '.npz'
    metadata = np.load(filename2, allow_pickle=True)    # load metadata from .npy file\
    history = metadata.f.arr_0
    scale_type = metadata.f.arr_1
    try:
        scale_params = metadata.f.arr_2
    except:
        scale_params = None
    try:
        freqlims = metadata.f.arr_3
    except:
        freqlims = None
    history = np.atleast_1d(history)[0]             # to solve that histroy is saved as a 0d numpyarray with a dictionary instead of a dictionary
    print('metadata loaded from', filename2)
    return NeuralNetwork(model, history=history, scale_type=scale_type, scale_params=scale_params, freqlims=freqlims)    # save as NeuralNetwork instance



def get_data(directory='data/H', verbose=True):
    """
    Retrieves spectral data from fits file and puts it into a numpy array.
    Observations from different polarizations need to be processed separately.
    It outputs two numpy arrays of the frequency and the flux, the HDU list, a list of subbands used and the length of these subbands. 
    The arrays are four-dimensional and have the following dimensions:
    1: observation number
    2: subband (4 at most in the case of HIFI, 3 in the case of e.g. band 7b)
    3: sub-observation or whatever that's called
    4: channels
    """
    # list files
    files = glob.glob('{}/*.fits'.format(directory))    # load fits files used for training the neural network
    files = list(files)             # in case there is only one file, make "files" into a list
    
    # in case whole filename given
    if type(directory) == str:
        if directory[-5:] == '.fits':
            files = [directory]
    print('files used:', files)
            
    # get data from files
    freq, flux = [], []
    file_len = []                   # list of number of spectra in each fits file
    for file in files:
        with fits.open(file) as hdul:
            # find size of data
            subbands = []           # list of non-empty subbands
            sub_len = []            # lengths of non-empty subbands
            for i in range(4):      # loop over subbands to find their shapes
                try:
                    N3_sub = hdul[1].header['SUBLEN{}'.format(i+1)]     # length of subband
                    if N3_sub > 0:
                        sub_len.append(N3_sub) 
                        subbands.append(i+1)    # add subband to list of non-empty subbands
                except:             # subband does not exist
                    pass

            N1 = len(subbands)                  # size of axis 1 (number of subbands)
            N2 = hdul[1].header['NAXIS2']       # size of axis 2
            N3 = np.max(sub_len)                # size of axis 3 (largest length of subbands)

            
            # put data into numpy arrays 
            freq_file, flux_file  = np.zeros((2, len(hdul), N1, N2, N3))        # make empty arrays
            for obs in range(len(hdul)-1):     # loop through table, first HDU is empty
                data = hdul[obs+1].data

                freq_obs, flux_obs = np.empty((2, N1, N2, N3))      # frequencies and fluxes contained in single HDU
                freq_obs[:], flux_obs[:] = np.infty, np.infty       # fill array with infinite values
                for i, sub in enumerate(subbands):
                    freq_sub = data['frequency_{}'.format(sub)]             # frequency in one subband
                    flux_sub = data['flux_{}'.format(sub)]                  # flux in one subband
                    freq_obs[i, :, :np.shape(freq_sub)[-1]] = freq_sub
                    flux_obs[i, :, :np.shape(flux_sub)[-1]] = flux_sub

                freq_file[obs] = freq_obs
                flux_file[obs] = flux_obs
                
        freq.append(freq_file)
        flux.append(flux_file)
        # number of spectra (divided by number of subbands)
        file_len.append(int(np.product(np.shape(flux_file)[:-1]) / np.shape(flux_file)[-3]))   
        if verbose: print('loaded', file)
            
    # replace infty values with nan
#     freq = [np.where(f == np.infty, np.nan, f) for f in freq]
#     flux = [np.where(f == np.infty, np.nan, f) for f in flux]
    return Data(freq, flux, hdul, subbands, sub_len, file_len)


def scale(X, scale_type='minmax_zero', scale_params=None):
    """Scale the data. This speeds up the training process."""
    X_f = X[np.isfinite(X)]
    if scale_type == 'normal':
        """Center around mean and divide by 6 times standard deviation."""
        p = [np.mean(X_f), 6*np.std(X_f)] if scale_params == None else scale_params
        X_s = (X - p[0]) / p[1]

    elif scale_type == 'minmax':
        """Set minimum to 0 and maximum to 1."""
        p = [np.min(X_f), None] if scale_params == None else scale_params
        X_s = X - p[0]
        p[1] = np.max(X_f) if scale_params == None else scale_params[1]
        X_s /= p[1]
    
    elif scale_type == 'minmax_zero' or scale_type == 'minmax_zero_01':
        """Set minimum to roughly -1 and maximum to roughly 1, with exact mean 0."""
#         p = [np.min(X_f), np.max(X_f), None] if scale_params == None else scale_params
#         X_s = X / ((p[1] - p[0]) / 2)
#         X_f = X_s[np.isfinite(X_s)]
#         p[2] = np.nanmean(X_f) if scale_params == None else scale_params[2]
#         X_s -= p[2]
        p = [None, None] if scale_params == None else scale_params
        try:
            try:
                X_f = X.data[~X.mask]
            except:
                X_f = X
            p[0] = np.nanmean(X_f) if scale_params == None else scale_params[0]
            X_s = X - p[0]
            try:
                X_f = X_s.data[~X.mask]
            except:
                X_f = X_s
            p[1] = np.nanmax(np.abs(X_f))
            X_s /= p[1]
            
            if scale_type == 'minmax_zero_01':
                X_s /= 2
                X_s += 1
        
        except ValueError:  # raised if X is empty.
            return X, p
        
    elif scale_type == 'none':
        """No scaling."""
        p = None

    return X_s, p

    
def rescale(x, scale_type, scale_params):
    """Undo the scaling of the data."""
    p = scale_params
    X = x
    if scale_type == 'normal':
        return X * p[1] + p[0]
    elif scale_type == 'minmax':
        return X * p[1] + p[0]
    elif scale_type == 'minmax_zero':
#         return (X + p[2]) * ((p[1] - p[0]) / 2)
        return X * p[1] + p[0]
    elif scale_type == 'minmax_zero_01':
        return 2*(X-1) * p[1] + p[0]
    elif scale_type == 'none':
        return X

    
def get_filenames(index=None, directory='nets', common='ann_', filetype='hdf5'):
    """
    Get available filenames to save a neural network model and its metadata.
    filename2 will be a .npz file to save the metadata such as history.
    """
    if index == None:
        index = get_filename_index(directory=directory, common=common)
    filename1 = '{}/{}{:04d}.{}'.format(directory, common, index, filetype)
    filename2 = '{}/{}{:04d}'.format(directory, common, index)
    return filename1, filename2

    
def get_filename_index(directory='nets'):
    """Find the highest index to get a unique filename"""
    files = glob.glob('{}/*'.format(directory))     # list files in directory
    search = [re.search('{}/(.*)_(.*).(h5|hdf5)'.format(directory), f, re.IGNORECASE) for f in files]     # search for .h5 and .hdf5 files
    n = [-1]            # list of file indices of files
    for s in search:
        if s is not None:
            n.append(int(s.group(2)))   # add file index to n
    index = max(n)+1    # use max index + 1 for new filename
    return index


def log(index, logdir='log.txt', **kwargs):
    """
    Write keywords and their values to a log file to keep track of the neural networks. 
    Index should be given to distinguish each log entry. This could be an int or str.
    """
    with open(logdir, 'a') as file:
        try:
            file.write('Neural network {}:\n'.format(index) + '-'*20 + '\n')
            for key, val in kwargs.items():
                # simplify float numbers
                try:
                    if type(val) != int: val = '{:.3e}'.format(val) 
                except:
                    pass
                
                # write entry to log file
                try:
                    file.write('{}:\t{}\n'.format(key, val)) 
                except:
                    pass
            file.write('\n\n')
        except:
            print('could not write info to log file')

        
def distribute_sets(n_samples, test_ratio=.2, seed=0):
    """
    Distribute the data over the training and testing sets.
    Returns training and testing booleans.
    """
    rn.seed(seed)   # set seed for reproducibility
    train = np.ones(n_samples, dtype=bool)
    train[:int(n_samples*test_ratio)] = 0
    rn.shuffle(train)
    test = ~train
    return train, test


def wT_lowfreq(arr):
    """
    Find lowfrequency structure in spectra. Wipe out any structure less than 8 channel frequency.
    Do not use this if channel is not evenly spaced.
    """
    coeff = pywt.wavedec(arr, 'db5', 'constant', 5)
    coeff[1][:] = 0.
    coeff[2][:] = 0.
    coeff[3][:] = 0.
    coeff[4][:] = 0.
    coeff[5][:] = 0.
    arr_wT = pywt.waverec(coeff, 'db5', 'constant')
    return arr_wT


def rebin(data, length, axis=1, func=np.mean):
    """Rebin an array along a given axis."""
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    binstep = dims[axis] / length
    data = data.transpose(argdims)
    data = [func(np.take(data, np.arange(int(i*binstep), int(i*binstep+binstep)), 0), 0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data


def congrid(x, n_resized, axis=1):
    n_rebin = np.shape(x)[axis]
    sampling_rebinned = np.linspace(0, 1, n_rebin)
    sampling_original = np.linspace(0, 1, n_resized)
    interpX = interp1d(sampling_rebinned, x, axis=axis)
    X_upscaled = interpX(sampling_original)
    return X_upscaled


def make_mock(n_mock_samples, n_channels, n_features_max=4, unity=False, max_width=1/5, return_params=False, datatype=np.float16):
    """
    Make mock features in the form of asymmetric Gaussians with varying amplitude, standard deviation and skewness.
    n_mock_samples:     number of spectra to generate (without noise and baseline)
    n_channels:         number of channels in a spectrum
    n_features_max:     maximum number of features in a single spectrum (number of features is randomly generated for each spectrum)
    unity:              if true, spectra will be normalized to 1, 
                        if false, sources will have amplitudes between 0.2 and 1 (two features at the same place can add up to larger values)
    max_width           maximum signal width as fraction of number of channels
    """
    mock = np.zeros((n_mock_samples, n_channels), dtype=datatype)
    std_list = []
    pos_list = []
    for i, mock_i in enumerate(mock):        
        n_features = rn.randint(1, 1+n_features_max)    # number of signals (randint(1, 2) gives only 1, so add 1 to n_features_max)
        signal_shift = np.clip(rn.normal(0, n_channels/6, n_features), -n_channels/2, n_channels/2)     # mean position of the signal (not accounting for skewness)
        amplitude = rn.uniform(0.2, 1, n_features)      # amplitude of the signals
        std = np.clip(np.abs(rn.normal(0, n_channels*max_width/3, n_features)), 1, n_channels*max_width)    # standard deviation
        alpha = rn.uniform(-3, 3, n_features)           # determines skewness
        
        std_list.append(std)
        pos_list.append(signal_shift + n_channels/2)
        for k in range(n_features):
            x = np.linspace(-5, 5, 10*std[k])
            signal_c = skewnorm.pdf(x, alpha[k])
            signal_c = signal_c / np.max(signal_c) * amplitude[k]     # rescale signal
            signal = np.zeros(n_channels)
            signal[int(n_channels/2 - len(signal_c)/2):int(n_channels/2 + len(signal_c)/2)] = signal_c
            signal = shift(signal, (signal_shift[k],))
            signal = datatype(signal)                   # reduce storage size of array
            mock_i += signal
        if unity:
            mock_i /= datatype(np.max(mock_i))
        mock[i] = mock_i
    if return_params: return mock, [std_list, pos_list]
    return mock


def add_mock(X, n_samples, n_channels, n_rebin, n_mps=1, mock_dir=None, SNR=False, n=None, max_SNR=5):
    """
    Add mock signals and noise to baselines (smoothed X), and rebins X to size n_rebin. IMPORTANT: either give mock_dir or SNR and n.
    X:          spectra used as baseline and to generate representative noise
    mock_dir:   directory containing the mock signals of size n_rebin
    SNR:        exact signal to noise ratio of each signal (should not be used for generating training data, only for chi squared testing)
    n:          number of spectra to generate (only if SNR is given)
    max_SNR:    maximum signal to noise ratio (not used if SNR is given)
    """
    X_smooth = wT_lowfreq(X)[:, :n_channels]        # get baselines
#     X_smooth_rebinned = rebin(X_smooth, n_rebin)    # rebin baselines to smalller size
#     X_rebinned = rebin(X, n_rebin)

    # number of spectra
    if n == None: 
        if type(SNR) is bool:
            n = n_samples*n_mps
        else:
            n = len(SNR)
            
    # replicate the noise from the data
    noise_X = X - X_smooth    # noise in the data
    std_noise = np.std(noise_X, axis=1)         # standard deviation of the noise in each spectrum
    amp_noise_mock = rn.normal(np.mean(std_noise), np.std(std_noise), (n, 1))    # randomized standard deviation of the noise
    noise_mock = rn.normal(0, 1, (n, n_channels))    # random noise
    noise_mock *= amp_noise_mock                # scale noise by a randomized standard deviation
    noise_mock_rebinned = congrid(noise_mock, n_rebin)
    print('generated mock noise')

    # rebin and add rebinned mock signals
    if type(SNR) is bool:
        """Load signals from source file. Note that the amplitudes of the signals can still be random."""
        amp_signal_mock = rn.uniform(0.2, 1, (n_mps*n_samples, 1)) * max_SNR * np.mean(amp_noise_mock)     # maximum amplitude of a single signal
        signal_mock = np.load(mock_dir)[:n]
        X_smooth = np.tile(X_smooth, (n_mps, 1))
    else:
        """Generate signals of amplitude 1 and multiply with SNR * noise to get signal with exact SNR. Generating signals takes some time."""
        SNR = SNR[..., np.newaxis]
        amp_signal_mock = SNR * amp_noise_mock                  # precise amplitude of a single signal 
        amp_signal_mock = amp_signal_mock[:n]
        signal_mock = np.load(mock_dir)[:n]
#         signal_mock = make_mock(n_mock_samples=n, n_channels=n_rebin, n_features_max=1, unity=True)

        # if number of requested spectra is larger than number of baselines (X_smooth), tile X_smooth
        s = np.shape(X_smooth)[0]
        if n > s:
            assert X_smooth.ndim == 2
            X_smooth = np.tile(X_smooth, (int(np.ceil(n/s)), 1))
        X_smooth = X_smooth[:n]
    signal_mock *= amp_signal_mock
    signal_mock = congrid(signal_mock, n_channels)
    """Get rid of negative values that emerge when using congrid. WARNING: will delete any absorption line mock signals."""
    signal_mock = np.where(signal_mock < 0, 0, signal_mock)
    print('generated mock signals')
    
    # add noise and signals to baseline
    X_mock = X_smooth + signal_mock + noise_mock
    print('mock dataset built')
    return X_mock.astype('float32'), X_smooth.astype('float32'), signal_mock.astype('float32'), noise_mock.astype('float32')


def predict(X, f1, f2, pca_dir, scale_type='minmax_zero', n_channels_x=None, DNN=True, spline=False):
    """Give the baseline prediction in the data X, using the neural networks with directories f1 and f2, and pca transformation."""
    assert X.ndim == 2, 'Input data should be 2-dimensional: (observations, channels).'
        
    # load networks and pca transformation
    if type(f1) == int:
        nn1 = load_NN(f1)
    else:
        nn1 = load_NN(fullpathname=f1)
    if type(f2) == int:
        nn2 = load_NN(f2)
    elif type(f2) == bool:
        nn2 = None
    else:
        nn2 = load_NN(fullpathname=f2)
    with open(pca_dir, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

    global _nn1, _nn2, _pca
    _nn1 = nn1; _nn2 = nn2; _pca = pca
    
    n_rebin = np.array(nn1.model.input_shape)[1]
    n_channels = np.shape(pca.components_)[1]
    n_channels_x = np.shape(X)[1]
    X0, scale_params = scale(X, scale_type)                 # scale data for faster trainingX1 = wT_lowfreq(X0)
    X1 = wT_lowfreq(X0)                                     # smooth with wavelet filter
    X1 = rebin(X0, n_rebin)[..., np.newaxis]              # reduce size of data
    X_cnn = nn1.model.predict(X1)[..., 0]                   # prediction of the CNN
    X_cnn_rescaled = congrid(X_cnn, n_channels)             # scale the output of the CNN to match the size of the eigenvectors

#     try:     # use clipped range if applicable (discard channels outside range)
#         clip_range = nn2.clip_range
#         X_cnn_rescaled = X_cnn_rescaled[:, clip_range[0]:clip_range[1]]
#     except:
#         pass
#     if n_channels_x == None: _, n_channels_x = X_cnn_resc.shape[-1]
    baseline_mean = np.mean(X_cnn_rescaled, axis=1)[..., np.newaxis]
    X_cnn_rescaled_zero = X_cnn_rescaled - baseline_mean
    if spline == True:
        splines = np.asarray([UnivariateSpline(np.arange(len(Xs)), Xs, k=5) for Xs in X_cnn_rescaled_zero])
        for i, Xs in enumerate(X_cnn_rescaled_zero):
            X_cnn_rescaled_zero[i] -= splines[i](np.arange(len(Xs)))
    
    if DNN == True:
        X_eigenvalues = pca.transform(X_cnn_rescaled_zero)      # transform to eigenvalues
        Y = nn2.model.predict(X_eigenvalues)[..., 0]            # prediction of the dense NN
    if DNN == False:
        Y = pca.inverse_transform(pca.transform(X_cnn_rescaled_zero))
        
    if spline == True:
        for i, y in enumerate(Y):
            Y[i] += splines[i](np.arange(len(y)))
    
    Y = Y + baseline_mean
    Y = congrid(Y, n_channels_x)                            # resize the prediction to match the original size of the spectra
    Y = rescale(Y, scale_type, scale_params)                # rescale the prediction back to original units
    return Y


# def chi_square(O, E, axis=1):
#     """Chi squared between observed (O) and expected (O) value."""
#     return np.sum(((O - E)**2 / (E+eps)), axis=axis)


def gaussian(x, x0, a, sigma):
    """Returns a Gaussian function to use for fitting."""
    return a * np.exp(-(x-x0)**2 / (2*sigma**2))


def gaussian_fixed_x0(x, a, sigma):
    """Returns a Gaussian function to use for fitting, but has fixed location at _xmid (should be global variable)."""
    return gaussian(x, _xmid, a, sigma)


def gaussian_fixed_x0_a(x, sigma):
    """Returns a Gaussian function to use for fitting, but has fixed location at _xmid and fixed amplitude _amp (should be global variables)."""
    return gaussian(x, _xmid, _amp, sigma)


def fitGaussian(y, signal):
    """
    Fit Gaussians to the data y and the signal. Returns fit parameters for data and signal.
    
    The Gaussians fitted to the data have fixed positions (only amplitude and sigma are fit parameters).
    The Gaussians fitted to the signal have fixed positions and amplitude, only sigma is a fit parameter.
    """
    global _xmid
    global _amp
    x = np.linspace(0, 1, n_channels_x)                 # x-range used for fitting, between 0 and 1
    _Xmid = np.argmax(signal, axis=1) / n_channels_x    # position of signals
    _Amp = np.max(signal, axis=1)                       # amplitudes of signals
    
    fits_guess = []     # fit parameters for y
    fits_truth = []     # fit parameters for the signal
    
    # loop over data
    for i, yi in enumerate(y):
        # change the global variables of the position and amplitude (these are implicitly used for the fit)
        _xmid = _Xmid[i]
        _amp = _Amp[i]
        
        # fit
        try:
            fit_guess, _ = curve_fit(gaussian_fixed_x0, x, yi, p0=[1, .05], bounds=(0, [10, .5]))
        except:
            fit_guess = [np.nan, np.nan]
        fit_truth, _ = curve_fit(gaussian_fixed_x0_a, x, signal[i], p0=.05, bounds=(0, .5))
        fits_guess.append(fit_guess)
        fits_truth.append(fit_truth)

    return np.transpose(fits_guess), np.transpose(fits_truth)


def explained_variance(eigvals):
    """For a given set of eigenvalues, returns an array with the explained variance for each eigenvector."""
    var = np.var(eigvals, axis=0)
    var_total = np.sum(var)
    return var/var_total


def notebook():
    pass
#     """Run %matplotlib notebook a couple of times because it doesn't just work in one time. Makes sure plots are run in notebook mode (interactive)."""
#     for i in range(100):
#         %matplotlib notebook        
        