from eswremove import *     # import eswremove module

# data directories
polarisation = 'H'  # polarisation of the training/testing data
data_dir = '/net/virgo01/data/users/roybos/ESW/data/stability_v3/' + polarisation   # baselines used for training/testing
mock_dir = '/net/virgo01/data/users/roybos/ESW/mock_2048_n5.npy'            # mock signals (used for training purposes, size does not need to exceed n_rebin)
index = str(get_filename_index(directory=data_dir))                         # index used for saving PCA and NN
pca_dir = data_dir + '/pca_' + index
logdir = data_dir + '/log.txt'

# training parameters
n_mps = 28          # number of mock spectra per baseline
n_rebin = 2048       # size of the scaled down spectra used to train the CNN (preferably a power of 2)
n_components = 6   # number of components used for the PCA
"""
How to scale the data before training. Either one of:
    'normal':       from -5 std to 5 std, zero mean 
    'minmax':       from -1 to 1 (mean is not zero, not recommended for PCA)
    'minmax_zero':  from roughly -1 to 1, zero mean (training is less effective for spectra with weak ESWs)
    'none':         no scaling
"""
scale_type = 'minmax_zero_01'


# load data and simplify format to 2D numpy array
data = get_data(data_dir)           # make Data instance of the baseline data
data.merge(cutoff=True, freqpad=100)             # simplify format and merge spectra of all subbands together; if cutoff=True, some overlapping data is discarded
data.remove_empty(fillnan=True, delta=0.05)                 # discard empty spectra; if fillnan=True, NaN values are interpolated
scale_params = data.scale(scale_type, individual=False)      # change scaling of data before training

# remove the masked values and find the shape of the data
X = data.X
_, n_channels = np.shape(X)                     # detemine number of channels
X = X.data[~X.mask].reshape(-1, n_channels)     # remove masked values
n_samples, n_channels = np.shape(X)             # detemine number of baselines

# divide into test and training sets
train_untiled, test_untiled = distribute_sets(n_samples, 0.2)
train = np.tile(train_untiled, (n_mps))
test = np.tile(test_untiled, (n_mps))
X0 = X

# # add mock signals and reduce size of spectra
X1_full, X_smooth, signal_mock, noise_mock = add_mock(X0, n_samples, n_channels, n_rebin, n_mps, mock_dir, max_SNR=5)       # add mock signals

print('getting input and output data')
baseline_mean_y = np.mean(X_smooth, axis=1)[:, np.newaxis]
X_smooth_zero = X_smooth - baseline_mean_y     # smooth with zero mean baseline (for PCA)

X1 = congrid(X1_full, n_rebin)                       # reduce size of data to n_rebin
Y1 = congrid(X_smooth, n_rebin)[..., np.newaxis]



# obtain the pca transformation and save to pca_dir
pca = PCA(n_components)
pca.fit(X_smooth[train])                        # perform PCA on smooth baselines
with open(pca_dir, 'wb') as pickle_file:        # save PCA transformation
    pickle.dump(pca, pickle_file)
    print('PCA transformation saved to', pca_dir)

    

def get_cnn():
    """Get the model for the convolutional neural network. The model needs to be trained using train_cnn()."""
    # find necessary padding by taking the difference with a power of 2
    i = 4
    while i <= 14:
        if 2**i >= n_rebin:
            remainder = 2**i % n_rebin
            break
        i += 1
    if remainder % 2 != 0:  # if remainder odd, use different padding on each side
        padding = (int(np.floor(remainder/2)), int(np.ceil(remainder/2)))
    else:                   # if remainder even, use equal padding on each side
        padding = int(remainder/2)
    print('padding:', padding)
    
    # encoding
    inputs = keras.Input(shape=(n_rebin, 1))
    x = layers.ZeroPadding1D(padding=padding)(inputs)
    x = layers.Conv1D(64, 80, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2,padding='same')(x)
    x = layers.Conv1D(32, 25, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2,padding='same')(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2,padding='same')(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2,padding='same')(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    encoded1 = layers.MaxPooling1D(2,padding='same')(x)    

    # decoding
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoded1)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 25, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(1, 80, activation='sigmoid', padding='same')(x)
    decoded = layers.Cropping1D(cropping=padding)(x)

    # autoencoder
    autoencoder = keras.Model(inputs, decoded)
    return autoencoder



def train_cnn(epochs=40, batch_size=500, lr=0.001, index=get_filename_index(), freqlims=None):
    """Train the model given in get_cnn(). Logs some of the parameters to the log file."""
    # set model parameters
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss = keras.losses.mean_squared_error

    # get available filenames to save the model
    filename1, filename2 = get_filenames(directory=data_dir, common='cnn_', index=index)
    
    # list of callbacks used to monitor the training process (early stopping, reverting to best performing model)
    cb_early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-9, patience=5, mode='min', verbose=0)
    cb_checkpoint = callbacks.ModelCheckpoint(filename1, save_best_only=True, monitor='val_loss', mode='min')
    cb_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0002, mode='min')

    # compile the model
    model = get_cnn()
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    
    # train the neural network
    print('started training at', time.strftime("%H:%M:%S", time.localtime()))
    start = time.time()
    model.fit(x=X1[train].astype('float32'), y=Y1[train].astype('float32'),
              validation_data=(X1[test].astype('float32'), Y1[test].astype('float32')),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb_early_stopping, cb_plateau, cb_checkpoint])
    done = time.time()
    nn = NeuralNetwork(model, scale_params=scale_params)
    print('training finished in {:.2f} seconds'.format(done-start))
        
    # save neural network and append notes to log file
    nn.save(freqlims, None, filename2)
    try:
        log(index, logdir=logdir,
            loss=loss.__name__,
            final_loss=np.min(model.history.history['val_loss']),
            epochs=epochs,
            batch_size=batch_size,
            datasize=n_rebin,
            scale_type=scale_type,
            optimizer=optimizer,
            polarization=polarisation,
            n_components=n_components,
            network_type='CNN autoencoder',
            note='2048 channels, individually scaled data',
            location=data_dir,
            )
    except:
        print('could not write to log')
    return nn


nn1 = train_cnn(epochs=30, batch_size=500, index=int(index), lr=0.001, freqlims=data.freqlims)


    
# prediction from CNN
Y1_pred = nn1.model.predict(X1)[..., 0]
Y1_pred_upscaled = congrid(Y1_pred, n_channels)
baseline_mean = np.mean(Y1_pred_upscaled, axis=1)[:, np.newaxis]
Y1_pred_upscaled_zero = Y1_pred_upscaled - baseline_mean


# inputs for dense NN
X2 = pca.transform(Y1_pred_upscaled_zero)
Y2 = X_smooth_zero
eigenvalues = pca.transform(X_smooth_zero)
coef_max = np.max(np.abs(eigenvalues))                          # maximum eigenvalue (eigenvalues are divided by this value for faster training)
eigenvectors = pca.components_                          # eigenvectors of PCA to be used for training
pca_mean = np.tile(pca.mean_, (n_samples*n_mps, 1))     # mean of PCA
    
    
    
def get_dense(nPCA_comp, coef_max, eigenvecs, vec_mean):
    """
    Get the dense neural network model. This model needs to be trained using the train_dense() function.
    nPCAcomp:   number of PCA components
    coef_max:   maximum eigenvalue in the training data
    eigenvecs:  eigenvectors obtained from PCA
    vec_mean:   mean of eigenvectors
    """
    noisy_coef = keras.Input(shape=(nPCA_comp, 1))                      # pca coefficients used as input
    noisy_coef_norm = layers.Lambda(lambda x: x/coef_max)(noisy_coef)   # normalize coefficients by dividing by maximum
    
    # concatenate coefficients and negatives of coefficients before applying relu in next layer
    x = layers.Concatenate(axis=1)([noisy_coef_norm, -noisy_coef_norm])
    
    # dense layers
    initializer = keras.initializers.RandomNormal(mean=0., stddev=0.01)
    x = layers.Dense(nPCA_comp*8, activation='relu', kernel_initializer=initializer)(x)
    x = layers.UpSampling1D(size=4)(x)
    
    x = layers.Conv1D(nPCA_comp*8, 2, padding='same', activation='sigmoid', kernel_initializer=initializer)(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dense(nPCA_comp*8, activation='softmax', kernel_initializer=initializer)(x)
    x = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(x)   
    
    # split the positive and negative coefficients and add them
    xp, xn = Split()(x)
    x = layers.Add()([xp, -xn])
    x = layers.Add()([noisy_coef_norm, x])
    
    # inverse PCA transformation
    coef_predicted_real = layers.Lambda(lambda x: x*coef_max)(x)        # undo normalization
    pca_vecs = tf.constant(eigenvecs.astype('float32'))
    pca_mean = tf.constant(vec_mean.astype('float32'))
    dot_eigen = tf.tensordot(coef_predicted_real, pca_vecs, axes=(1,0))
    inverse_PCA = tf.math.add(dot_eigen, pca_mean)
    
#     initializer = keras.initializers.RandomNormal(mean=0., stddev=0.01)
#     x = layers.Dense(nPCA_comp, activation='tanh', kernel_initializer=initializer)(inverse_PCA)
#     x = layers.Dense(n_channels, activation='tanh', kernel_initializer=initializer)(x)
# #     x = tf.math.multiply(inverse_PCA, x)
#     x = tf.math.add(inverse_PCA, x)
# #     x = layers.Dense(n_channels, activation='tanh', kernel_initializer=initializer)(x)

    out = layers.Reshape((n_channels, 1))(inverse_PCA)
    NN = keras.Model(inputs=noisy_coef, outputs=out)

    NN.summary()
    return NN


def train_dense(epochs=50, batch_size=500, index=get_filename_index(), freqlims=None):
    """Train the model given in get_dense(). Logs some of the parameters to the log file."""
    # get available filenames to save the model
    filename1, filename2 = get_filenames(directory=data_dir, common='dnn_', index=index)

    # set model parameters
    optimizer = tf.keras.optimizers.Nadam(lr=0.01, beta_1=0.94, beta_2=0.999, epsilon=1e-07)
    loss = keras.losses.mean_squared_error

    # list of callbacks used to monitor the training process (early stopping, reverting to best performing model)
    cb_early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=4e-8, patience=50, mode='min', verbose=0)
    cb_checkpoint = callbacks.ModelCheckpoint(filename1, save_best_only=True, monitor='val_loss', mode='min')
    cb_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=2e-4, factor=0.5, patience=8, min_lr=0.002, mode='min')

    # compile the model
    model = get_dense(n_components, coef_max, eigenvectors, pca.mean_)
    model.compile(optimizer=optimizer, loss=loss)

    # train the neural network
    print('started training at', time.strftime("%H:%M:%S", time.localtime()))
    start = time.time()
    model.fit(x=X2[train], y=Y2[train],
              validation_data=(X2[test], Y2[test][..., np.newaxis]),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb_checkpoint, cb_plateau])
    done = time.time()
    print('training finished in {:.2f} seconds'.format(done-start))
    
    # save neural network and append notes to log file
    nn = NeuralNetwork(model, scale_params=scale_params)
    nn.save(freqlims, None, filename2)
    log(index, logdir=logdir,
        loss=loss.__name__, 
        final_loss=np.min(model.history.history['val_loss']), 
        epochs=epochs, 
        batch_size=batch_size, 
        datasize=n_channels,
        scale_type=scale_type,
        optimizer=optimizer,
        polarisation=polarisation,
        n_components=n_components,
        network_type='dense NN',
        note='Additional dense layers after transformation, individually scaled data, eigenvalues split into positive and negative values with relu',
       )
    return nn


nn2 = train_dense(epochs=10, batch_size=1000, index=int(index), freqlims=data.freqlims)