import os
import argparse
import ipdb
import scipy
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def argparser():
    parser = argparse.ArgumentParser(description='Normalization stats (z-score) for various preprocessed datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, help='path of the input dataset')
    parser.add_argument('-m', '--mode', required=True, choices=['mat_cwt'], help='mode of computation, determined by the type of input data')
    parser.add_argument('-o', '--output_filename', required=True, help='filename of the output .npz (without extension)')
    parser.add_argument('-f', '--n_features', default=19, type=int, help='number of features of the StandardScaler (i.e. number of output normal distributions)')
    args = parser.parse_args()
    return args


def feature_matrix_mat_cwt(file_path: str, n_features: int) -> npt.NDArray:
    """
    Receive the path to a single .mat file of a CWT with shape
    (n_freq, n_time_samples, n_electrodes) and compute the feature matrix
    used to run the partial_fit() of a StandardScaler.
    The feature matrices have shape (n_samples, n_features), so here
      - n_samples = pixels of the cwt of a single electrode (2D image)
      - n_features = number of electrodes
    We want a normal distribution for each electrode.
    """
    cwt = scipy.io.loadmat(file_path)['cwts']
    matr = np.reshape(cwt, (-1, n_features))
    return matr


def main():
    args = argparser()
    full_path = os.path.join(os.path.expanduser(args.path), 'data')
    file_list = sorted(os.listdir(full_path))
    scaler = StandardScaler()

    n_iter = len(file_list)
    for i in tqdm(range(n_iter), ncols=100):
        current_file = os.path.join(full_path, file_list[i])
        # The partial_fit() of the StandardScaler accepts matrices
        # with shape (n_samples, n_features). The number of features is
        # the number of output normal distributions.
        if args.mode == 'mat_cwt':
            feature_matrix = feature_matrix_mat_cwt(current_file, args.n_features)
            scaler.partial_fit(feature_matrix)

    list_mean = scaler.mean_
    list_std = np.sqrt(scaler.var_)
    np.savez(os.path.join('local', args.output_filename + '.npz'), list_mean=list_mean, list_std=list_std)

    print('\nNormalization stats saved to disk. Exiting...')


if __name__ == "__main__":
    main()
