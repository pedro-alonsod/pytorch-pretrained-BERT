import subprocess, sys
from pathlib import Path
import numpy as np

seq_lengths = [128, 205, 282, 359, 435, 512]
batch_sizes = [12, 17, 22, 27, 32]
epochs = [1, 3, 5, 7, 10]
num_runs = 3
seeds = [42, 45892, 17485]

for epoch in epochs:
    folder_location = f'/nfs/staff/jacnil/askreddit-bert-base-test/val/ep-{epoch}--bs-27--sl-359'
    for run, seed in zip(range(num_runs), seeds):
        subprocess.call(['python', './run_classifier.py', '--data_dir', '../data', '--bert_model', 'bert-base-uncased', '--task_name', 'reddit', '--output_dir', f'{folder_location}/run-{run}', '--do_train', '--do_eval', f'--num_train_epochs={epoch}', '--train_batch_size', '27', '--max_seq_length', '359', '--seed', f'{seed}'])
