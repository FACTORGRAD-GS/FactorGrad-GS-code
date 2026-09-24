#!/usr/bin/env python3
"""Create a TensorBoard viewer from the sampled exact scalar CSV export."""
import argparse
import csv
import lzma
from torch.utils.tensorboard import SummaryWriter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('csv_xz')
parser.add_argument('--output',required=True)
args=parser.parse_args()
with lzma.open(args.csv_xz,'rt',newline='') as stream,SummaryWriter(args.output) as writer:
    for row in csv.DictReader(stream):
        writer.add_scalar('event_'+row['event_file']+'/'+row['tag'],float(row['value']),
                          global_step=int(row['step']),walltime=float(row['wall_time_s']))
