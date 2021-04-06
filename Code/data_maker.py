import scipy.io as sio
import numpy as np

def make_data(mat_file_address, crop_length):
    ## loading the .mat data files
    crop_length = int(crop_length)
    data_file = sio.loadmat(mat_file_address)
    names = []
    for x in data_file:
        print(x)
        if x not in ['__header__', '__version__', '__globals__']:
            names.append(x)
    print('following files will be processed', names)
    for x in names:
        data = data_file[x]
        print('shape of data is ', np.shape(data))
        data = data[crop_length:np.shape(data)[0] - crop_length, crop_length:np.shape(data)[0] - crop_length, :]
        [m, n, f] = np.shape(data)
        print('shape of ', x, ' data is ', m, 'x', n, 'x', f)

        ## preprocessing data before analysis.
        # 1- the camera can capture cosmic ray background which introduces spikes at random pixels in each frame.
        # spikes can be removed by ignoring pixels that are a few standard deviations away from mean of each frame.
        # spikes can be both positive or negative.
        stdN = 5  # number of standard deviations allowed.
        print('removing spikes from ', x)
        for c in range(f):
            print('frame', c)
            data_std = np.std(data[:, :, c])
            data_mean = np.mean(data[:, :, c])
            for a in range(m):
                for b in range(n):
                    if np.abs(data[a, b, c]) > (stdN * data_std):
                        data[a, b, c] = data_mean
        print('spikes removed in ', x)

        # shifting and renormalizing

        print('adjusting offset to bring min value to zero and normalizing with max value')

        data_spectra = []
        for a in range(m):
            for b in range(n):
                spec = data[a, b, :] - min(data[a, b, :])
                if max(spec) > 0:
                    spec = spec / max(spec)
                    data_spectra.append(spec)
        print('max value of ', x, '_spectra is ', np.max(data_spectra))
        print(x, ' adjusted and renormalized')

        print(x, ' spectrum', data_spectra[500])

        print('shape of ', x, '_spectra is', np.shape(data_spectra))

        print('saving ', x, ' spectra')
        name_path = '/home/shah/PycharmProjects/NNs for Chemistry/processed data/New multi sams data-stamped_' + x
        np.save(name_path, data_spectra)

    return data_spectra
