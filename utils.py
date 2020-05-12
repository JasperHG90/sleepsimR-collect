#
# Utility functions used to download sleep data from Physionet
#   database can be found here: https://physionet.org/content/sleep-edfx/1.0.0/
#
# Please also see the following resources on which the functions below are based:
#   BACKGROUND INFORMATION & TUTORIALS
#   - https://raphaelvallat.com/bandpower.html
#   - https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html#sphx-glr-auto-tutorials-sample-datasets-plot-sleep-py
#   - https://www.mdpi.com/1099-4300/18/9/272/htm
#   FILTERING AND PREPROCESSING
#   - https://mne.tools/dev/auto_tutorials/discussions/plot_background_filtering.html#disc-filtering
#   - https://mne.tools/dev/auto_tutorials/intro/plot_10_overview.html#preprocessing
#   - https://mne.tools/dev/generated/mne.filter.filter_data.html
#   - https://mne.tools/dev/auto_tutorials/preprocessing/plot_30_filtering_resampling.html#tut-filter-resample
#
# Written by: Jasper Ginn <j.h.ginn@students.uu.nl>
#

# For sleep stages
import mne
from mne.filter import filter_data
from mne.time_frequency import psd_welch, psd_multitaper, csd_fourier, tfr_morlet
# Misc
import os
import numpy as np
import urllib.request
from typing import Tuple
import json
# To get index data
from collections import defaultdict
from bs4 import BeautifulSoup
from requests import get
import re

# Logger
import logging
import daiquiri

# Set up logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

### Load data into global environment

# Load frequency bands
with open("frequency_bands.json", "r") as inFile:
    frequency_bands = json.load(inFile)

# Load sleep data index
# Get sleepfile index
def get_sleepfile_index():

    """
    Retrieves an index with the sleep files if it does not yet exist in the directory
    """

    # Url containing index files
    DATA_URL = "https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/#files-panel"
    # Read HTML
    BODY = get(DATA_URL)
    # Use beautifulsoup to parse HTML
    soup = BeautifulSoup(BODY.text, "html.parser")

    # Search in the HTML for the table containing links etc.
    files = soup.find("div", attrs = {"id": "files-panel"}).find("table", attrs = {"class": "files-panel"}).find("tbody").find_all("tr")
    # Remove first element (irrelevant link)
    del files[0]
    # To store data
    names_and_links = {}
    for file in files:
        ele = file.find_all("td")
        # Get file name
        file_name = ele[0].find("a").text
        # File recording
        file_download = "https://physionet.org" + ele[0].find("a", attrs = {"class":"download"}).get("href")
        # Data type
        nsplit = file_name.strip("\\.edf").split("-")
        dtype = nsplit[-1]
        # Subject ID
        id_and_recording = re.search("^[A-Z]{2}4([0-9]{3})", nsplit[0]).group(1)
        subject_id = id_and_recording[:2]
        subject_recording = id_and_recording[2]
        # Make hashmap
        if names_and_links.get(int(subject_id)) is None:
            names_and_links[int(subject_id)] = {1: {"PSG": {}, "Hypnogram": {}}, 2: {"PSA": {}, "Hypnogram": {}}}
        # Populate
        names_and_links[int(subject_id)][int(subject_recording)][dtype] = {"file_name": file_name, "file_url": file_download}

    # Save
    with open("sleepdata_index.json", "w") as outFile:
        json.dump(names_and_links, outFile)

# Check if index file present. Else download
if not "sleepdata_index.json" in os.listdir():
    logger.info("No file index found. Downloading now ...")
    get_sleepfile_index()
# Load index
with open("sleepdata_index.json", "r") as inFile:
    sleepdata_index = json.load(inFile)

# Turn keys into integers
sleepdata_index = {int(k):{int(a):b for a,b in v.items()} for k,v in sleepdata_index.items()}

# Other settings
# This is a mapping from original sleep data to integer-based mapping
sleep_stage_mapping = {'Sleep stage W': 1,
                      'Sleep stage 1': 2,
                      'Sleep stage 2': 3,
                      'Sleep stage 3': 4,
                      'Sleep stage 4': 4,
                      'Sleep stage R': 5}

# Sleep stages where NREM stages are unified
nrem_unified_mapping = {'Sleep stage W': 1,
                        'Sleep stage NREM': 2,
                        'Sleep stage 2': 3,
                        'Sleep stage 3/4': 4,
                        'Sleep stage R': 5}

### Error class

class ProcessingError(Exception):
    pass

### Utility functions

def download_sleepdata(subject_id: int, recording: int, folder: str) -> Tuple[str, str]:

    """
    Downloads the hypnogram and EEG data for the subject

    :param subject_id: subject id
    :param recording: either 1 or 2 value indicating the night of the recording to download
    :param folder: folder in which to store downloaded data

    :returns: tuple containing (hypnogram_location, PSG_location)
    """

    # Subject id must be between 1 and 82
    assert type(subject_id) == int and subject_id >= 0 and subject_id <= 82, "Subject ID must be an integer between 1 and 82 (inclusive)"
    assert type(recording) == int and recording >= 1 and recording <= 2, "Recording must be either '1' (first night) or '2' (second night)"

    # Assert folder exists
    assert os.path.exists(folder), "Folder '{}' does not exist".format(folder)

    # Get data record for subject
    data_record = sleepdata_index.get(subject_id)
    if data_record is None:
        return(None, None)
    data_record = data_record.get(recording)
    # If recordings are empty, then return none
    if data_record["Hypnogram"].get("file_url") is None or data_record["PSG"].get("file_url") is None:
        return(None, None)
    # Extract
    HYP_url = data_record["Hypnogram"]["file_url"]
    subject_HYP = data_record["Hypnogram"]["file_name"]
    PSG_url = data_record["PSG"]["file_url"]
    subject_PSG = data_record["PSG"]["file_name"]

    # ID to character
    if subject_id < 10:
        subject_id = "0" + str(subject_id)

    # Local files
    PSG_local = os.path.join(folder, subject_PSG)
    HYP_local = os.path.join(folder, subject_HYP)

    # If exists, move on
    if not os.path.isfile(PSG_local):
        urllib.request.urlretrieve(PSG_url, PSG_local)
    else:
        logger.info("PSG data for subject '{}', recording '{}' already exists. Moving on ...".format(subject_id, recording))
    if not os.path.isfile(HYP_local):
        urllib.request.urlretrieve(HYP_url, HYP_local)
    else:
        logger.info("HYP data for subject '{}', recording '{}' already exists. Moving on ...".format(subject_id, recording))

    # Return
    return(HYP_local, PSG_local)

def preprocess_sleepdata(path_to_PSG: str, path_to_HYP: str, folder_out:str, epoch_length = 30):

    """
    Preprocess the sleep data of a subject
    """

    # Get user id
    UID = path_to_PSG.split("/")[-1].split("-")[0]

    # Preprocessed data file name
    data_out_filename = os.path.join(folder_out, "{}-preprocessed.npy".format(UID))
    tpm_out_filename = os.path.join(folder_out, "{}-tpm.npy".format(UID))

    # Read raw data
    PSG_raw = mne.io.read_raw_edf(path_to_PSG, preload=False, exclude=["Temp rectal", "Resp oro-nasal",
                                                                         "EMG submental", "Event marker"])

    # Read annotations (sleep stages)
    HYP_raw = mne.read_annotations(path_to_HYP)

    # Add sleep stage annotations to the data
    PSG_raw.set_annotations(HYP_raw, emit_warning=False)

    # Re-label sleep stages from 6 to 3 possible stages
    HYP_preprocess, _ = mne.events_from_annotations(PSG_raw,event_id=sleep_stage_mapping,chunk_duration=30.)

    # Possible other preprocessing steps

    # Create 30-second epochs
    tmax = 30. - 1. / PSG_raw.info['sfreq']  # tmax in included
    try:
        PSG_epoched = mne.Epochs(raw=PSG_raw, events=HYP_preprocess,
                                  event_id=nrem_unified_mapping, tmin=0., tmax=tmax,
                                  baseline=None)
    except ValueError:
        raise ProcessingError("Problem occurred while epoching data ...")
    # Power spectrum analysis
    welch_out = welch_power_band(PSG_epoched, use_channels = ["eeg", "eog"])
    # Save events and processed data
    subject_events = PSG_epoched.events[:,2].reshape(welch_out.shape[0], 1)
    # Find out where subject is not awake
    not_awake = np.where((subject_events != 1) == True)[0]
    # First event marks the first time a user is not awake
    # Last event marks the last time user is not awake
    # Subset data in a window around these events
    # A = first time - 50 epochs
    # B = A + 2 * 30 * 60 * 8 = A + 28800 gives an 8-hour window
    # NB: the assumption is that the subject does not take naps
    idx_start = not_awake[0] - 200
    idx_end = idx_start + (12 * 60 * 2) # H * (epochs * 2)
    # Concatenate sleep stages with preprocessed data
    data_out = np.concatenate([subject_events, welch_out], axis=1)
    # Index for 8-hour window
    data_out = data_out[idx_start:idx_end, :]
    # Make transition probability matrix based on sleep states
    try:
        user_tpm = make_tpm(subject_events[idx_start:idx_end,:])
        np.save(tpm_out_filename, user_tpm)
    except IndexError:
        logger.error("Could not make TPM")
    # Index and save data
    np.save(data_out_filename, data_out)

def welch_power_band(data_preprocessed, normalize = True, use_channels = ["eeg"]):

    """
    Use Welch' algorithm to preprocess the EEG/EOG channels in the data

    :seealso: - documentation for psd_welch ==> https://mne.tools/stable/generated/mne.time_frequency.psd_welch.html
              - MNE tutorial on sleep staging ==>
    """
    # See doc for psd_welch: https://mne.tools/stable/generated/mne.time_frequency.psd_welch.html
    psds, freqs = psd_multitaper(data_preprocessed, picks= use_channels, fmin=1, fmax=45.)
    # If norm
    if normalize:
        # Normalize the PSDs
        psds /= np.sum(psds, axis=-1, keepdims=True)
    # 'psds' is of shape (epochs, channels, n_segments)
    Xmu = []
    for freq_min, freq_max in frequency_bands.values():
        # For each EEG/EOG channel, compute the frequency bands
        for i in range(psds.shape[1]):
            Xmu.append(np.median(psds[:, i, (freqs >= freq_min) & (freqs < freq_max)], axis=1).reshape(psds.shape[0], 1))
            Xmu.append(np.mean(psds[:, i, (freqs >= freq_min) & (freqs < freq_max)], axis=1).reshape(psds.shape[0], 1))
            Xmu.append(np.var(psds[:, i, (freqs >= freq_min) & (freqs < freq_max)], axis=1).reshape(psds.shape[0], 1))
            Xmu.append(np.min(psds[:, i, (freqs >= freq_min) & (freqs < freq_max)], axis=1).reshape(psds.shape[0], 1))
            Xmu.append(np.max(psds[:, i, (freqs >= freq_min) & (freqs < freq_max)], axis=1).reshape(psds.shape[0], 1))
        # Extract the
        # psds_band = psds[:, :, (freqs >= freq_min) & (freqs < freq_max)]
    # Stack for each frequency and each channel column-wise and return
    return(np.concatenate(Xmu, axis=1))

def make_tpm(subject_events: np.array):
    """
    Given some annotated sleep data, create a ground-truth transition probability matrix
    """
    # Get unique sleep stages
    m = len(np.unique(subject_events))
    # Make numpy array
    tpm = np.zeros((m, m))
    # For each step in the sleep stages, add 1 to tpm
    # Save the first event here
    state_prev = subject_events[0,:].squeeze()
    for step in range(1, subject_events.shape[0]):
        # Get current state
        state_current = subject_events[step, :].squeeze()
        # Add to tpm
        tpm[state_prev-1, state_current-1] += 1
        # Current is previous for next one
        state_prev = state_current
    # Divide by row sums
    for i in range(m):
        tpm[i,:] /= sum(tpm[i,:])
    # Round
    tpm = np.round(tpm, 4)
    # Return
    return(tpm)
