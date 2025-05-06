# spectrogram_utils.py
import os
import numpy as np
import shutil
import librosa
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.filters import gabor_kernel
from collections import OrderedDict
import argparse
import ast


def compute_spectrogram(window_size=1024,
                        percent_overlap=50,
                        window_type=('tukey', 0.25),
                        scaling='density',
                        mode='psd',
                        ):

    # Expect the sampling rate to be the same for all recordings.
    sampling_rate_expected = 44100

    # The window overlap for the spectrogram.
    window_overlap = int(percent_overlap * window_size / 100)

    # The window used for the spectrogram. ('tukey', 0.25) is the default.

    # boxcar
    # triang
    # blackman
    # hamming
    # hann
    # bartlett
    # flattop
    # parzen
    # bohman
    # blackmanharris
    # nuttall
    # barthann
    # cosine
    # exponential
    # tukey
    # taylor
    # lanczos

    window_type = signal.get_window(window_type, window_size)
    window_description = f"{window_type[0]}({window_type[1]})" if isinstance(
        window_type, tuple) else window_type

    cwd = os.getcwd()
    # The directory where the .wav files are located. We will read these.
    recording_dir = cwd + '/Recordings/'

    # The directory where we will write the computed spectrograms as .png image files.
    spectrogram_dir = cwd + '/Spectrograms/'

    os.makedirs(spectrogram_dir, exist_ok=True)

    # Write spectrogram parameters.
    parameters_filename = spectrogram_dir + '0_spectrogram_parameters.txt'

    # Write the parameters to a file.
    with open(parameters_filename, 'w') as spectogram_param_file:
        spectogram_param_file.write(
            f'sampling rate = {sampling_rate_expected}\n')
        spectogram_param_file.write(f'window size = {window_size}\n')
        spectogram_param_file.write(f'window overlap = {window_overlap}\n')
        spectogram_param_file.write(f'window overlap = {window_description}\n')
        spectogram_param_file.write('log10 of spectrum\n')

    # Get list of recordings.
    recording_list = os.listdir(recording_dir)
    total_recordings = len(recording_list)
    print(f"[Spectrogram] Found {total_recordings} recordings to process.")
    recording_num = 0
    for recording in recording_list:
        recording_num += 1
        recording_filename = recording_dir + recording 
        print(f"[Spectrogram] Processing {recording_num}/{total_recordings}: {recording}")
        # Read the audio file. Don't change the sample rate.
        x, sampling_rate = librosa.load(recording_filename, sr=None)

        if sampling_rate != sampling_rate_expected:
            print(f'WARNING: {recording} has wrong sampling rate: expected=' +
                  f'{sampling_rate_expected} recording={sampling_rate}')
            break

        # Compute the spectrogram.
        w, t, s = signal.spectrogram(x,
                                     sampling_rate,
                                     window=window_type,
                                     noverlap=window_overlap,
                                     nperseg=window_size,
                                     scaling=scaling,
                                     mode=mode
                                     )

        # s: matrix with the 2D spectrogram (will be complex).
        # w: vector with the frequencies spacings of the computed DFT.
        # t: vector with the time spacings of the windows.

        # Compute the spectrum of the spectrogram.
        s_spectrum = abs(s)

        # Flip it vertically.
        s_spectrum = np.flip(s_spectrum, 0)

        # Compute log_10 of spectrogram.
        s_spectrum_log = np.log10(s_spectrum + np.finfo(float).eps)
        # plt.imshow(s_spectrum_log,cmap='gray')

        # Normalize so values range from 0 to 1.
        s_spectrum_log = (s_spectrum_log - np.amin(s_spectrum_log)) / \
            (np.amax(s_spectrum_log) - np.amin(s_spectrum_log))

        # Save as png file.
        spectrogram_filename = spectrogram_dir + recording + '.png'
        plt.imsave(spectrogram_filename, s_spectrum_log, cmap='gray')

        # Save time and frequency indices.
        spectrograminfo_filename = spectrogram_dir + recording + '_info.txt'
        spectrograminfo_file = open(spectrograminfo_filename, 'w')
        for i in t:
            spectrograminfo_file.write(str(i)+'\n')
        for i in w:
            spectrograminfo_file.write(str(i)+'\n')
        spectrograminfo_file.close()

    return True


def compute_rois():

    cwd = os.getcwd()

    # root_dir of where to put annotated spectrograms, etc.
    root_dir = cwd+'/ROIs/'

    # Where the computed spectrograms are located.
    spectrogram_dir = cwd+'/Spectrograms/'

    # Where the ROI .csv files are located.
    rois_dir = cwd+'/ROIs/'

    # The .csv file with the info about the ROIs.
    rois_info_file = cwd+'/ROIs/birds.csv'

    # Read the ROI info.
    rois_info = pd.read_csv(rois_info_file)

    n_birds = 5
    print(f"[ROIs] Starting ROI extraction for {n_birds} birds.")
    # For each bird, read the ROI file and process the ROIs.
    for i in range(n_birds):
        bird = rois_info['bird'][i]
        print(f"[ROIs] Bird {i+1}/{n_birds}: {bird}")
        # Read ROI .csv file for this bird.
        roi_csv_file = rois_dir + rois_info['roi_file'][i]
        bird_rois_info = pd.read_csv(roi_csv_file)
        n_rois = rois_info['roi_count'][i]
        # Process each ROI for this bird.
        for j in range(n_rois):
            recording = bird_rois_info['recording'][j]
            print(f"[ROIs]   ROI {j+1}/{n_rois} for recording {recording}")
            # Always get original spectroram.
            spectrogram = recording + '.png'
            spectrogram_filename = spectrogram_dir + spectrogram
            spectrogram_image = plt.imread(spectrogram_filename)

            # Check if this spectrogram has already been marked with an ROI.
            spectrogram_roi = recording + '.roi.png'
            spectrogram_roi_filename = root_dir + \
                rois_info['bird'][i] + '/' + spectrogram_roi

            if os.path.exists(spectrogram_roi_filename):
                # Read already marked up spectrogram.
                spectrogram_roi_image_PIL = Image.open(
                    spectrogram_roi_filename)
            else:
                os.makedirs(root_dir + rois_info['bird'][i], exist_ok=True)
                spectrogram_roi_image_PIL = Image.open(spectrogram_filename)

            # Get size of spectrogram.
            n_rows = np.shape(spectrogram_image)[0]
            n_cols = np.shape(spectrogram_image)[1]

            # Get time and frequency coordinates of spectrogram from _info.txt
            spectrogram_info = recording + '_info.txt'
            spectrogram_info_filename = spectrogram_dir + spectrogram_info
            spectrograminfo_file = open(spectrogram_info_filename, 'r')

            time_coords = np.zeros(n_cols)
            for x in range(n_cols):
                time_coords[x] = spectrograminfo_file.readline()
            freq_coords = np.zeros(n_rows)
            for x in range(n_rows):
                freq_coords[x] = spectrograminfo_file.readline()
            spectrograminfo_file.close()
            # Reverse.
            freq_coords = np.flip(freq_coords)

            # Find pixel bounds of ROI.
            top_left_row = np.where(freq_coords < bird_rois_info['y2'][j])
            bottom_right_row = np.where(freq_coords < bird_rois_info['y1'][j])
            top_left_col = np.where(time_coords > bird_rois_info['x1'][j])
            bottom_right_col = np.where(time_coords > bird_rois_info['x2'][j])
            width = bottom_right_col[0][0] - top_left_col[0][0] + 1
            height = bottom_right_row[0][0] - top_left_row[0][0] + 1

            # Draw rectangle.
            img1 = ImageDraw.Draw(spectrogram_roi_image_PIL)
            img1.rectangle([(top_left_col[0][0], top_left_row[0][0]),
                           (bottom_right_col[0][0], bottom_right_row[0][0])], outline='white')

            # Save spectrogram with marked ROIs.
            spectrogram_roi_image_PIL.save(spectrogram_roi_filename)

            # Extract and save ROI.
            image_roi = spectrogram_image[top_left_row[0][0]-1:bottom_right_row[0]
                                          [0]-1, top_left_col[0][0]-1:bottom_right_col[0][0]-1]
            image_roi_filename = root_dir + \
                rois_info['bird'][i]+'/ROIs/'+rois_info['bird'][i] + \
                '_'+str(bird_rois_info['id'][j])+'.png'

            if os.path.exists(root_dir+rois_info['bird'][i]+'/ROIs/'):
                plt.imsave(image_roi_filename, image_roi, cmap='gray')
            else:
                try:
                    os.makedirs(root_dir+rois_info['bird'][i]+'/ROIs/')
                except Exception as e:
                    print(f"Error creating directory: {e}")
                plt.imsave(image_roi_filename, image_roi, cmap='gray')


def compute_mean_std_features():
    print("[MeanStd] Starting mean/std feature computation")
    cwd = os.getcwd()

    # Create folders if they don't exist.
    try:
        os.makedirs(cwd+'/features')
        os.makedirs(cwd+'/images')
    except Exception as e:
        print(f"Error creating directory: {e}")

    # Logic to copy ROI's calculated to one folder named 'image' for feature extraction
    if len(os.listdir(cwd+'/images')) == 0:
        for i in os.listdir(cwd+'/ROIs'):
            if os.path.isdir(cwd+'/ROIs/'+i):
                if len(os.listdir(cwd+'/ROIs/'+i+'/ROIs/')) == 0:
                    print("Folder of ROI empty")
                else:
                    for j in os.listdir(cwd+'/ROIs/'+i+'/ROIs/'):
                        shutil.copy(cwd+'/ROIs/'+i+'/ROIs/'+j, cwd+'/images')
    else:
        print("Files exist in image folder")

    # Location of images.
    image_dir = cwd + '/images/'

    # Location of where to write features.
    feature_dir = cwd + '/features/'

    # The .csv file containing the image names and classes.
    image_file = cwd + '/image_names_classes.csv'

    # Number of images.
    n_images = 218

    # Read image names and classes .csv file.
    image_names_classes = pd.read_csv(image_file, header=None)
    # print(image_names_classes[1][0])

    # Mean + standard deviation features have dimension 2
    fdim = 2

    features = np.zeros((n_images, fdim))

    # Extract features std_mean for each image.
    for i in range(n_images):

        # Read the image.
        filename = image_dir + image_names_classes[0][i]
        print(f"[MeanStd] Processing image {i+1}/{n_images}: {filename}")
        im = plt.imread(filename)

        # It turns out that the spectrogram images saved using plt.imsave have four channels
        # RGBA. The RGB channels are each equal to the grayscale value so we can use any of them.

        # Compute the mean of the grayscale image as the first feature dimension.
        features[i, 0] = np.mean(im[:, :, 0])

        # Compute the standard deviation of the grayscale image as the second feature dimension.
        features[i, 1] = np.std(im[:, :, 0])

    feature_filename = feature_dir + 'mean_stdev.txt'

    np.savetxt(feature_filename, features, delimiter=',')


def compute_gabor_texture_features():
    print("[Gabor] Starting Gabor texture feature computation")
    cwd = os.getcwd()

    # Create folders if don't exist.
    try:
        os.makedirs(cwd+'/features')
        os.makedirs(cwd+'/images')
    except Exception as e:
        print(f"Error creating directory: {e}")

    # Logic to copy ROI's calculated to one folder named 'image' for feature extraction
    if len(os.listdir(cwd+'/images')) == 0:
        for i in os.listdir(cwd+'/ROIs'):
            if os.path.isdir(cwd+'/ROIs/'+i):
                if len(os.listdir(cwd+'/ROIs/'+i+'/ROIs/')) == 0:
                    print("Folder of ROI empty")
                else:
                    for j in os.listdir(cwd+'/ROIs/'+i+'/ROIs/'):
                        shutil.copy(cwd+'/ROIs/'+i+'/ROIs/'+j, cwd+'/images')
    else:
        print("Files exist in image folder")

    # Location of images.
    image_dir = cwd + '/images/'

    # Location of where to write features.
    feature_dir = cwd + '/features/'

    # The .csv file containing the image names and classes.
    image_file = cwd + '/image_names_classes.csv'

    # Number of images.
    n_images = 218

    # Read image names and classes .csv file.
    image_names_classes = pd.read_csv(image_file, header=None)

    # Create Gabor filter bank kernels.
    kernels = []

    # EXPERIMENT WITH THE NUMBER OF ORIENTATIONS AND SCALES.
    # This code will create a feature file with the name Gabor_Y_X_.txt
    # where X is the number of scales and Y number of orientations.

    nscales = 3
    norientations = 4

    min_frequency = 0.05
    max_frequency = 0.4

    for orientation in range(norientations):
        theta = (orientation / norientations) * np.pi
        for scale in range(nscales):
            frequency = min_frequency + scale * \
                ((max_frequency - min_frequency)/(nscales-1))
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)
            # print('orientation='+str(orientation)+': theta='+str(theta))
            # print('scale='+str(scale)+': frequency='+str(frequency))

    # Visualize the filters as images. Note that the filters have different sizes--see the axes in the plots below.
    # The filters are complex so first visualize the real parts and then the imaginary parts.
    fig, axs = plt.subplots(1, len(kernels), figsize=(20, 20))
    for k, kernel in enumerate(kernels):
        axs[k].imshow(np.real(kernel), cmap='gray')

    fig, axs = plt.subplots(1, len(kernels), figsize=(20, 20))
    for k, kernel in enumerate(kernels):
        axs[k].imshow(np.imag(kernel), cmap='gray')

    # Mean and standard deviation will be computed for each filter output.
    fdim = 2 * len(kernels)

    features = np.zeros((n_images, fdim))

    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels) * 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = signal.convolve(image, kernel)
            feats[2*k] = np.abs(filtered).mean()
            feats[2*k+1] = np.abs(filtered).std()
        return feats

    # Extract features for each image.
    for i in range(n_images):

        # Read the image.
        filename = image_dir + image_names_classes[0][i]
        print(f"[Gabor] Processing image {i+1}/{n_images}: {filename}")
        im = plt.imread(filename)

        # It turns out that the spectrogram images saved using plt.imsave have four channels
        # RGBA. The RGB channels are each equal to the grayscale value so we can use any of them.

        features[i, :] = compute_feats(im[:, :, 0], kernels)

    # Save the features as a .csv file.
    gabor_filename = 'gabor_results.txt'
    feature_filename = feature_dir + gabor_filename

    np.savetxt(feature_filename, features, delimiter=',')


def perform_query(window_size=1024,
                  percent_overlap=50,
                  window_type=('tukey', 0.25),
                  scaling='density',
                  mode='psd',):
    print("[Query] Starting retrieval experiments")
    # Feature type.

    cwd = os.getcwd()

    # The filename with the features to use (including the folder).
    feature_filename = cwd+'/features/gabor_results.txt'
    # print("Feature filename: " + feature_filename)

    # Read the features.
    features = np.genfromtxt(feature_filename, delimiter=',')

    fdim = np.shape(features)[1]
    print('features have dimension:' + str(fdim))

    # Number of images.
    n_images = 218

    # Read image names and classes .csv file.
    # The .csv file containing the image names and classes.
    image_file = cwd + '/image_names_classes.csv'
    image_names_classes = pd.read_csv(image_file, header=None)

    # Dictionary to store precision and recall for the queries performed.
    precision_recall = OrderedDict()

    # Dictionary to store the class and class size of birds.
    bird_class_size_dict = OrderedDict()

    # Where the ROI .csv files are located.
    rois_dir = cwd + '/ROIs/'

    # The .csv file with the info about the ROIs.
    rois_info_file = cwd+'/ROIs/birds.csv'
    rois_info = pd.read_csv(rois_info_file)

    # Store the class and class size of birds.
    for i in range(len(list(rois_info['bird']))):
        bird_class_size_dict[i+1] = rois_info['roi_count'].iloc[i]

    # Number of images.
    n_images = 218

    def L2_dist(feature1, feature2):
        # Get dimension.
        dim = np.shape(feature1)[0]

        result = 0
        for k in range(dim):
            result = result + ((feature1[k] - feature2[k])**2)
        result = result ** 0.5

        return (result)

    def L1_dist(feature1, feature2):
        # Get dimension.
        dim = np.shape(feature1)[0]

        result = 0
        for k in range(dim):
            result = result + abs(feature1[k] - feature2[k])

        return (result)

    # Perform the retrievals using each image as a query.
    for query_image in range(n_images):
        print(f"[Query] Query {query_image+1}/{n_images}")
        # Compute distance between query feature vector and every target image's feature vector (including the query).
        distances = np.zeros(n_images)

        for j in range(n_images):
            # Can choose difference distance measures;
    #        distances[j] = L1_dist(features[query_image], features[j])
            distances[j] = L2_dist(features[query_image], features[j])
    #        print('distances['+str(j)+'] = '+str(distances[j]))

        # Get the indices of the sorted distances.
        sorted_index = np.argsort(distances)

        # Get the class and class size of query image
        class_qimg = image_names_classes[1][query_image]
        class_qimg_size = bird_class_size_dict[class_qimg]

        temp_prec_arr = []
        temp_recal_arr = []

        # Compute precision and recall for different number of retrieved images.
        for k in range(1, n_images+1):
            prec_num = 0
            prec_denom = k
            recal_denom = class_qimg_size
            recal_num = 0
            for z in sorted_index[0:k]:
                if class_qimg == image_names_classes[1][z]:
                    prec_num += 1
                    recal_num += 1
            temp_prec_arr.append(prec_num/prec_denom)
            temp_recal_arr.append(recal_num/recal_denom)

            precision_recall[query_image] = [temp_prec_arr, temp_recal_arr]

    # Interpolate precision for fixed recall values.

    pre_defined_recall = [0.0, 0.1, 0.2, 0.3,
                          0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Empty dictionary to hold the interpolated values.
    precision_recall_inter = OrderedDict()

    # Function to return the interpolated precision value

    def interpolate_p_r(p_arr, r_arr):
        temp_p = []
        for i in pre_defined_recall:
            for j in range(len(r_arr)):
                if r_arr[j] >= i:
                    temp_p.append(p_arr[j])
                    break
        return temp_p

    # Interpolate precision values at fixed recall values.
    for k, v in precision_recall.items():
        ans = interpolate_p_r(v[0], v[1])
        precision_recall_inter[k] = ans

    # Average the interpolated precision values over all queries.
    average_prec = np.zeros(len(pre_defined_recall))
    for k, v in precision_recall_inter.items():
        average_prec = np.add(average_prec, v)
    for i in range(len(average_prec)):
        average_prec[i] = average_prec[i]/n_images

    print(average_prec)

    # Create precision_recall folder if it doesn't exist.
    try:
        os.makedirs(cwd+'/precision_recall')
    except Exception as e:
        print(f"Error creating directory: {e}")

    # Save the average precision values as a .csv file.
    fname = f"Gabor_3_4_{window_size}_{percent_overlap}_{window_type}_{scaling}_{mode}"
    average_precision_filename = cwd+'/precision_recall/' + fname + '_p_r.txt'

    np.savetxt(average_precision_filename, average_prec, delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute spectrograms, ROIs, features and run queries"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1024,
        help="Length of each spectrogram segment (nperseg)"
    )
    parser.add_argument(
        "--percent_overlap",
        type=float,
        default=50,
        help="Overlap between segments, as percentage of window_size (0–100)"
    )
    parser.add_argument(
        "--window_type",
        type=str,
        default="('tukey', 0.25)",
        help="Window type tuple, e.g. \"('tukey', 0.25)\" or a single string like \"hann\""
    )
    parser.add_argument(
        "--scaling",
        type=str,
        choices=['density', 'spectrum'],
        default='density',
        help="Spectrogram scaling: 'density' (PSD per Hz) or 'spectrum' (total power)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['psd', 'magnitude'],
        default='psd',
        help="Spectrogram mode: 'psd' or 'magnitude'"
    )
    args = parser.parse_args()

    # Convert the window_type string into a tuple or string
    try:
        window_type = ast.literal_eval(args.window_type)
    except Exception:
        window_type = args.window_type

    compute_spectrogram(
        window_size=args.window_size,
        percent_overlap=args.percent_overlap,
        window_type=window_type,
        scaling=args.scaling,
        mode=args.mode
    )
    compute_rois()
    compute_gabor_texture_features()
    perform_query()
