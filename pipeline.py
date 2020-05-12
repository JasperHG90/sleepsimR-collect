#
# Script used to download and preprocess sleepdata
#
# Written by: Jasper Ginn
#

import argparse
from utils import *
import os
import logging
import daiquiri

# Set up logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

if __name__ == "__main__":

    # Get arguments
    parser = argparse.ArgumentParser(description='Download and preprocess sleep data')
    # Add the arguments
    parser.add_argument('subject_ids',
                        type=str,
                        help='Subject ids for which to download and preprocess data. You may either (1) pass a string of comma-separated integers (e.g. "1,2,3, ...") or (2) pass a range by passing e.g. "1:8" for the first 8 subjects. Note that the number of subjects runs from 0 to 82. I adopt the convention here that these subject ids range from 1 to 83')
    parser.add_argument('recording',
                        type=int,
                        choices = [1, 2],
                        help='Sleep recording for night (1) or night (2) for each of the above subjects.')
    parser.add_argument('raw_data_path',
                        type=str,
                        help="Path used to store the downloaded data.")
    parser.add_argument('processed_data_path',
                        type=str,
                        help="Path used to store the preprocessed data.")
    # Get arguments
    args = parser.parse_args()
    # Get subject ids
    subject_ids = args.subject_ids
    recording = args.recording
    # Hold final ids
    final_ids = []
    # Check if commas present
    if "," in subject_ids:
        # Split
        subject_ids = [id.strip() for id in subject_ids.split(",")]
    else:
        subject_ids = [subject_ids]
    # If colon present ...
    for subject_id in subject_ids:
        if ":" in subject_id:
            # Create range
            r_in = subject_id.split(":")
            r_out = [i for i in range(int(r_in[0])-1, int(r_in[1]))]
            # Add to final ids
            for id_out in r_out:
                # Assert in range
                if not (id_out >= 0 and id_out <=82):
                    logger.error("Subject ids must range from 1 to 83")
                    raise ValueError("Subject ids must range from 1 to 83")
                final_ids.append(id_out)
        else:
            subject_id = int(subject_id) - 1
            # Assert in range
            if not (subject_id >= 0 and subject_id <=82):
                logger.error("Subject ids must range from 1 to 83")
                raise ValueError("Subject ids must range from 1 to 83")
            final_ids.append(subject_id)
    # Download files
    FOLDER_RAW = args.raw_data_path
    FOLDER_PROCESSED = args.processed_data_path
    if not os.path.isdir(FOLDER_RAW):
        os.makedirs(FOLDER_RAW)
    if not os.path.isdir(FOLDER_PROCESSED):
        os.makedirs(FOLDER_PROCESSED)
    # Preprocess files
    logger.info("Now downloading and preprocessing data for {} subjects ...".format(len(final_ids)))
    for subject_id in final_ids:
        # Download files
        logger.info("Downloading files for subject {} ...".format(subject_id + 1))
        HYP_subj, PSG_subj = download_sleepdata(subject_id, recording = recording, folder = FOLDER_RAW)
        # If None, continue
        if HYP_subj is None or PSG_subj is None:
            logger.info("Data for subject id '{}', recording '{}' does not exist. Skipping these recordings ...".format(subject_id, recording))
            continue
        # Preprocess files
        logger.info("Preprocessing files for subject {} ...".format(subject_id))
        try:
            preprocess_sleepdata(PSG_subj, HYP_subj, FOLDER_PROCESSED)
        except ProcessingError:
            logger.error("Problem occurred with subject '{}', recording '{}'".format(subject_id, recording))
            continue
    # Exit
    logger.info("Finished downloading and preprocessing data for {} subjects. Exiting now ...".format(len(final_ids)))
