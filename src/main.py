# spectrogram_utils.py
import os
import numpy as np
import shutil
import librosa
import pywt
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.filters import gabor_kernel
from collections import OrderedDict
import argparse
import ast
import psutil, os
import json

# Add Beanie and Motor imports
from beanie import Document, init_beanie
import motor.motor_asyncio
import asyncio

# global verbose flag
verbose = False

def log(msg):
    if verbose:
        print(msg)


def compute_spectrogram(window_size=1024,
                        percent_overlap=50,
                        window_type=('tukey', 0.25),
                        scaling='density',
                        mode='psd',
                        resample_method='none',
                        target_sr=44100):
    # Expect the sampling rate to be the same for all recordings.
    sampling_rate_expected = target_sr

    # The window overlap for the spectrogram.
    window_overlap = int(percent_overlap * window_size / 100)

    # The window used for the spectrogram. ('tukey', 0.25) is the default.

    # boxcar
    # blackman
    # hamming
    # hann
    # blackmanharris
    # nuttall

    window_type = signal.get_window(window_type, window_size)
    window_description = f"{window_type[0]}({window_type[1]})" if isinstance(
        window_type, tuple) else window_type

    cwd = os.getcwd()
    # The directory where the .wav files are located. We will read these.
    recording_dir = cwd + '/Recordings/'

    # Get list of recordings.
    recording_list = os.listdir(recording_dir)
    total_recordings = len(recording_list)
    log(f"[Spectrogram] Found {total_recordings} recordings to process.")
    recording_num = 0
    # Instead of saving to disk, store spectrogram images in RAM
    spectrogram_images = {}
    spectrogram_info = {}
    for recording in recording_list:
        recording_num += 1
        recording_filename = recording_dir + recording 
        log(f"[Spectrogram] Processing {recording_num}/{total_recordings}: {recording}")
        # Read the audio file, applying the requested resample_method.
        if resample_method == 'none':
            x, sampling_rate = librosa.load(recording_filename, sr=None)
        if resample_method == 'librosa':
            # simple librosa load+resample
            x, sampling_rate = librosa.load(recording_filename, sr=target_sr)
        elif resample_method == 'polyphase':
            # high-quality band-limited sinc (polyphase) resampling
            x, orig_sr = librosa.load(recording_filename, sr=None)
            x = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr, res_type='soxr_hq')
            sampling_rate = target_sr
        else:
            # none or naive (integer decimation)  
            x, orig_sr = librosa.load(recording_filename, sr=None)
            if resample_method == 'naive':
                x = x[::2]
                sampling_rate = orig_sr // 2
            else:
                sampling_rate = orig_sr

        if sampling_rate != sampling_rate_expected and resample_method != 'none':
            print(f'WARNING: {recording} has wrong sampling rate: expected=' +
                    f'{sampling_rate_expected} recording={sampling_rate}')
            break

        # Compute the spectrogram.
        w, t, s = signal.spectrogram(x,
                                     fs=sampling_rate,
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

        # Store in RAM instead of saving as PNG
        spectrogram_images[recording] = s_spectrum_log
        spectrogram_info[recording] = {'t': t, 'w': w}

        # Remove file I/O for images and info
        # plt.imsave(spectrogram_filename, s_spectrum_log, cmap='gray')
        # with open(spectrograminfo_filename, 'w') as spectrograminfo_file:
        #     for i in t: spectrograminfo_file.write(str(i)+'\n')
        #     for i in w: spectrograminfo_file.write(str(i)+'\n')

    # Add this to define spectrogram_parameters in memory
    spectrogram_parameters = {
        "sampling_rate": sampling_rate_expected,
        "window_size": window_size,
        "window_overlap": window_overlap,
        "window_description": window_description,
        "log": "log10 of spectrum"
    }

    return spectrogram_images, spectrogram_info, spectrogram_parameters


def compute_rois(spectrogram_images, spectrogram_info):

    cwd = os.getcwd()

    # Where the ROI .csv files are located.
    rois_dir = cwd+'/ROIs/'

    # The .csv file with the info about the ROIs.
    rois_info_file = cwd+'/ROIs/birds.csv'

    # Read the ROI info.
    rois_info = pd.read_csv(rois_info_file)

    # Add this line to initialize the dictionary
    roi_images = {}

    n_birds = 5
    log(f"[ROIs] Starting ROI extraction for {n_birds} birds.")
    # For each bird, read the ROI file and process the ROIs.
    for i in range(n_birds):
        bird = rois_info['bird'][i]
        log(f"[ROIs] Bird {i+1}/{n_birds}: {bird}")
        # Read ROI .csv file for this bird.
        roi_csv_file = rois_dir + rois_info['roi_file'][i]
        bird_rois_info = pd.read_csv(roi_csv_file)
        n_rois = rois_info['roi_count'][i]
        # Process each ROI for this bird.
        for j in range(n_rois):
            recording = bird_rois_info['recording'][j]
            log(f"[ROIs]   ROI {j+1}/{n_rois} for recording {recording}")
            # Always get original spectroram.
            spectrogram_image = spectrogram_images[recording]
            t = spectrogram_info[recording]['t']
            w = spectrogram_info[recording]['w']

            n_rows = np.shape(spectrogram_image)[0]
            n_cols = np.shape(spectrogram_image)[1]

            time_coords = np.array(t)
            freq_coords = np.array(w)
            freq_coords = np.flip(freq_coords)

            # Find pixel bounds of ROI.
            top_left_row = np.where(freq_coords < bird_rois_info['y2'][j])
            bottom_right_row = np.where(freq_coords < bird_rois_info['y1'][j])
            top_left_col = np.where(time_coords > bird_rois_info['x1'][j])
            bottom_right_col = np.where(time_coords > bird_rois_info['x2'][j])
            width = bottom_right_col[0][0] - top_left_col[0][0] + 1
            height = bottom_right_row[0][0] - top_left_row[0][0] + 1

            # Draw rectangle on a copy of the image (optional, for visualization)
            # Not needed if only extracting ROI

            # Extract and store ROI in RAM
            image_roi = spectrogram_image[top_left_row[0][0]-1:bottom_right_row[0][0]-1,
                                          top_left_col[0][0]-1:bottom_right_col[0][0]-1]
            roi_key = f"{rois_info['bird'][i]}_{bird_rois_info['id'][j]}"
            roi_images[roi_key] = image_roi

            # Remove file I/O for saving ROI images
            # plt.imsave(image_roi_filename, image_roi, cmap='gray')
    return roi_images


def compute_mean_std_features(roi_images):
    log("[MeanStd] Starting mean/std feature computation")
    # ...existing code...
    # Remove all file/folder creation and copying logic

    # The .csv file containing the image names and classes.
    cwd = os.getcwd()
    image_file = cwd + '/image_names_classes.csv'
    n_images = 218
    image_names_classes = pd.read_csv(image_file, header=None)

    fdim = 2
    features = np.zeros((n_images, fdim))

    for i in range(n_images):
        roi_key = image_names_classes[0][i].replace('.png', '')
        if roi_key in roi_images:
            im = roi_images[roi_key]
            log(f"[MeanStd] Processing image {i+1}/{n_images}: {roi_key}")
            features[i, 0] = np.mean(im)
            features[i, 1] = np.std(im)
        else:
            log(f"[MeanStd] Missing ROI image for {roi_key}")

    # Remove file writing
    # feature_filename = feature_dir + 'mean_stdev.txt'
    # np.savetxt(feature_filename, features, delimiter=',')
    return features


def compute_gabor_texture_features(roi_images):
    log("[Gabor] Starting Gabor texture feature computation")
    # ...existing code...
    # Remove all file/folder creation and copying logic

    cwd = os.getcwd()
    image_file = cwd + '/image_names_classes.csv'
    n_images = 218
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

    fdim = 2 * len(kernels)
    features = np.zeros((n_images, fdim))

    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels) * 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            # Ensure kernel is not larger than image
            if image.shape[0] < kernel.shape[0] or image.shape[1] < kernel.shape[1]:
                # Skip this kernel if it's too large for the image
                continue
            filtered = signal.convolve(image, kernel, mode='same')
            feats[2*k] = np.abs(filtered).mean()
            feats[2*k+1] = np.abs(filtered).std()
        return feats

    for i in range(n_images):
        roi_key = image_names_classes[0][i].replace('.png', '')
        if roi_key in roi_images:
            im = roi_images[roi_key]
            log(f"[Gabor] Processing image {i+1}/{n_images}: {roi_key}")
            features[i, :] = compute_feats(im, kernels)
        else:
            log(f"[Gabor] Missing ROI image for {roi_key}")

    # Remove file writing
    # gabor_filename = 'gabor_results.txt'
    # feature_dir = cwd + '/features/'
    # feature_filename = feature_dir + gabor_filename
    # np.savetxt(feature_filename, features, delimiter=',')
    return features


def perform_query(gabor_features,
                  window_size=1024,
                  percent_overlap=50,
                  window_type=('tukey', 0.25),
                  scaling='density',
                  mode='psd',
                  resample_method='none',
                  target_sr=44100,
                  mongodb_url="mongodb://mongo:27017"):
    log("[Query] Starting retrieval experiments")
    cwd = os.getcwd()
    image_file = cwd + '/image_names_classes.csv'
    image_names_classes = pd.read_csv(image_file, header=None)
    n_images = 218
    features = gabor_features

    # Dictionary to store precision and recall for the queries performed.
    precision_recall = OrderedDict()

    # Dictionary to store the class and class size of birds.
    bird_class_size_dict = OrderedDict()


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
        log(f"[Query] Query {query_image+1}/{n_images}")
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


    # Remove file writing for precision_recall
    params = {
        "window_size": window_size,
        "percent_overlap": percent_overlap,
        "window_type": str(window_type),
        "scaling": scaling,
        "mode": mode,
        "resample_method": resample_method,
        "target_sr": target_sr,
        "feature_type": "gabor",
        "gabor_nscales": 3,
        "gabor_norientations": 4
    }
    # Use asyncio to save to DB
    async def save_to_db():
        await init_db(mongodb_url)
        result_doc = ExperimentResult(parameters=params, results=average_prec.tolist())
        await result_doc.insert()
    asyncio.run(save_to_db())

    # Instead, return the average_prec array
    return average_prec


# Define Beanie Document for results and parameters
class ExperimentResult(Document):
    parameters: dict
    results: list

    class Settings:
        name = "experiment_results"



async def init_db(mongodb_url):
    client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
    await init_beanie(database=client.eecs207, document_models=[ExperimentResult])


def print_max_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Max RSS (resident set size): {mem_info.rss / (1024*1024):.2f} MB")


def run_experiment(
    window_size,
    percent_overlap,
    window_type,
    scaling,
    mode,
    resample_method,
    target_sr,
    mongodb_url,
    _verbose=False
):


    # Set global verbose flag
    global verbose
    verbose = _verbose

    # Main computation
    spectrogram_images, spectrogram_info, spectrogram_parameters = compute_spectrogram(
        window_size=window_size,
        percent_overlap=percent_overlap,
        window_type=window_type,
        scaling=scaling,
        mode=mode,
        resample_method=resample_method,
        target_sr=target_sr
    )
    roi_images = compute_rois(spectrogram_images, spectrogram_info)
    gabor_features = compute_gabor_texture_features(roi_images)
    average_prec = perform_query(
        gabor_features,
        window_size=window_size,
        percent_overlap=percent_overlap,
        window_type=window_type,
        scaling=scaling,
        mode=mode,
        resample_method=resample_method,
        target_sr=target_sr,
        mongodb_url=mongodb_url
    )
    
    print_max_memory_usage()
    print("Spectrogram parameters (in memory):", spectrogram_parameters)
    print("Gabor features shape:", gabor_features.shape)
    print("Average precision", average_prec)
    return average_prec



def main():
    parser = argparse.ArgumentParser(
        description="Compute spectrograms, ROIs, features and run queries"
    )
    parser.add_argument("--job_id", type=int, required=True, help="Job index (0-based)")
    parser.add_argument("--matrix_path", type=str, default="matrix.json", help="Path to matrix.json")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--mongodb_url", type=str, default="mongodb://mongo:27017", help="MongoDB connection URL")
    args = parser.parse_args()

    # set global verbose
    global verbose
    verbose = args.verbose

    # Load matrix
    with open(args.matrix_path, "r") as f:
        matrix = json.load(f)

    # Compute chunk indices
    chunk_size = 72
    start = args.job_id * chunk_size
    end = min(start + chunk_size, len(matrix))
    chunk = matrix[start:end]

    print(f"Job {args.job_id}: processing experiments {start} to {end-1}")

    for i, params in enumerate(chunk):
        # Unpack parameters, provide defaults if missing
        run_experiment(
            window_size=int(params.get("window_size", 1024)),
            percent_overlap=int(params.get("percent_overlap", 50)),
            window_type=params.get("window_type", ("tukey", 0.25)),
            scaling=params.get("scaling", "density"),
            mode=params.get("mode", "psd"),
            resample_method=params.get("resample_method", "none"),
            target_sr=int(params.get("target_sr", 44100)),
            mongodb_url=args.mongodb_url,
            _verbose=args.verbose
        )

if __name__ == "__main__":
    main()

