#!/usr/bin/env python
"""
Script to compute pooled EER for ASVspoof2021 DF. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py score.txt ./keys eval
"""
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob
import json

if len(sys.argv) != 4:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
truth_dir = sys.argv[2]
phase = sys.argv[3]

cm_key_file = os.path.join(truth_dir, 'CM/CustomDataset_metadata.txt')


def eval_to_score_file(score_file, cm_key_file):
    
    cm_data = pandas.read_csv(cm_key_file, sep='- ', header=None)
    cm_data[1] = cm_data[1].str.strip()
    cm_data[4] = cm_data[4].str.strip()
    
    submission_scores = pandas.read_csv(score_file, sep=' = ', header=None, skipinitialspace=True)
    submission_scores[0] = pandas.DataFrame([file.split('/')[-1] for file in list(submission_scores[0])])[0]
    
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)

    cm_scores = submission_scores.merge(cm_data[cm_data[5] == phase], left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
    bona_cm = cm_scores[cm_scores[4] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[4] == 'spoof']['1_x'].values

    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100*eer_cm)
    print(out_data)
    dic = {
        'bona_cm': str(bona_cm),
        'spoof_cm': str(spoof_cm),
        'EER': 100*eer_cm
    }
    with open('eer_auc_pretrained_custom_dataset.json', 'w') as f:
        json.dump(dic, f, indent=2)
    return eer_cm

if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)
        
    if not os.path.isdir(truth_dir):
        print("%s doesn't exist" % (truth_dir))
        exit(1)

    if phase != 'progress' and phase != 'eval' and phase != 'hidden_track':
        print("phase must be either progress, eval, or hidden_track")
        exit(1)

    _ = eval_to_score_file(submit_file, cm_key_file)
