import subprocess, sys
from pathlib import Path
import numpy as np

seq_lengths = [128, 205, 282, 359, 435, 512]
batch_sizes = [12, 17, 22, 27, 32]
num_runs = 3
seeds = [42, 45892, 17485]

for batch_size in batch_sizes:
    for seq_length in seq_lengths:
        folder_location = f'/nfs/staff/jacnil/askreddit-bert-base-test/val/bs-{batch_size}--sl-{seq_length}'
        for run, seed in zip(range(num_runs), seeds):
            subprocess.call(['python', './run_classifier.py', '--data_dir', '../data', '--bert_model', 'bert-base-uncased', '--task_name', 'reddit', '--output_dir', f'{folder_location}/run-{run}', '--do_train', '--do_eval', '--num_train_epochs=5', '--train_batch_size', f'{batch_size}', '--max_seq_length', f'{seq_length}', '--seed', f'{seed}'])
