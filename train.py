"""
Train a contrastive learning model to co-embed labelling patterns and chemical
structures of known metabolites, and make predictions for held-out peaks.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# add script directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets import IsotopicPatternDataset, collate_fn
from fingerprints import get_fingerprint
from functions import clean_mols
from losses import CLIPLoss
from loggers import EarlyStopping, log_loss, print_update
from models import Isopleth


# CLI
parser = argparse.ArgumentParser()

# fingerprint configuration
parser.add_argument('--fp_algo', type=str, required=True,
                    choices=['ecfp', 'fcfp'], default='ecfp')
parser.add_argument('--fp_type', type=str, required=True,
                    choices=['bits', 'counts'], default='bits')
parser.add_argument('--fp_length', type=int, required=True, default=2048)
parser.add_argument('--fp_radius', type=int, required=True, default=3)

# model architecture
parser.add_argument('--emb_dim', type=int, required=True, default=1024)
parser.add_argument('--fp_hidden_dim', type=int, required=True, default=1024)
parser.add_argument('--patt_hidden_dim1', type=int, required=True,
                    default=512)
parser.add_argument('--patt_hidden_dim2', type=int, required=True,
                    default=512)
parser.add_argument('--fp_dropout', type=float, default=0)
parser.add_argument('--patt_dropout', type=float, default=0)

# loss
parser.add_argument('--tau_clip', type=float, required=True, default=0.1)
parser.add_argument('--label_smoothing_clip', type=float, default=0.0)
parser.add_argument('--learnable_temperature', action='store_true')

# training
parser.add_argument('--batch_size', type=int, required=True, default=256)
parser.add_argument('--seed', type=int, required=True, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--ema_alpha', type=float, default=0.1)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--max_steps', type=int, default=int(1e9))
parser.add_argument('--max_epochs', type=int, default=int(1e9))
parser.add_argument('--log_every_steps', type=int, default=50)
parser.add_argument('--log_every_epochs', type=int, default=1)

# input/output files
parser.add_argument('--train_mat_file', type=str, required=True)
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--test_mat_file', type=str, required=True)
parser.add_argument('--test_cands_file', type=str, required=True)
parser.add_argument('--model_file', type=str, required=True)
parser.add_argument('--loss_file', type=str, required=True)
parser.add_argument('--pred_file', type=str, required=True)
parser.add_argument('--time_file', type=str, required=True)

args = parser.parse_args()
print(args)

# create config
config = Config(
    fp_algo=args.fp_algo,
    fp_type=args.fp_type,
    fp_length=args.fp_length,
    fp_radius=args.fp_radius,
    emb_dim=args.emb_dim,
    fp_hidden_dim=args.fp_hidden_dim,
    patt_hidden_dim1=args.patt_hidden_dim1,
    patt_hidden_dim2=args.patt_hidden_dim2,
    tau_clip=args.tau_clip,
    batch_size=args.batch_size,
    seed=args.seed,
    fp_dropout=args.fp_dropout,
    patt_dropout=args.patt_dropout,
    learning_rate=args.learning_rate,
    patience=args.patience,
    ema_alpha=args.ema_alpha,
    max_grad_norm=args.max_grad_norm,
    max_steps=args.max_steps,
    max_epochs=args.max_epochs,
    log_every_steps=args.log_every_steps,
    log_every_epochs=args.log_every_epochs,
    label_smoothing_clip=args.label_smoothing_clip,
    learnable_temperature=args.learnable_temperature
)

# seed all RNGs for reproducibility
seed = int(config.get_param('seed'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# create output directory if it does not exist
output_dir = os.path.dirname(args.model_file)
if output_dir and not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

# detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# start the timer
start_time = time.time()

# create dataset
train_dataset = IsotopicPatternDataset(
    config=config,
    label_file=args.train_mat_file,
    struct_file=args.train_file,
    device=device,
    seed=seed,
)

# initialize model
label_size = train_dataset.num_labels()
model = Isopleth(
    fp_in_dim=config.get_param('fp_length'),
    fp_hidden_dim=config.get_param('fp_hidden_dim'),
    fp_dropout=config.get_param('fp_dropout'),
    patt_in_dim=label_size,
    patt_hidden_dim1=config.get_param('patt_hidden_dim1'),
    patt_hidden_dim2=config.get_param('patt_hidden_dim2'),
    patt_dropout=config.get_param('patt_dropout'),
    emb_dim=config.get_param('emb_dim'),
)
model = model.to(device)
model.train()

# set up loss
clip_loss = CLIPLoss(
    temperature=config.get_param('tau_clip'),
    label_smoothing=config.get_param('label_smoothing_clip'),
    learnable_temperature=config.get_param('learnable_temperature'),
).to(device)

# set up data loader
# drop_last=True to avoid incomplete batches (CLIP loss requires >= 2)
loader = DataLoader(
    train_dataset,
    batch_size=int(config.get_param('batch_size')),
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)

# set up optimizer and early stopping
optim = Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08,
             lr=config.get_param('learning_rate'))
early_stop = EarlyStopping(patience=config.get_param('patience'),
                           ema_alpha=config.get_param('ema_alpha'))

train_start_time = time.time()

# training loop
counter = 0
for epoch in range(0, config.get_param('max_epochs')):
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        counter += 1
        if counter > config.get_param('max_steps'):
            break
        
        patterns, true_fps = batch
        
        fp_emb, patt_emb = model(true_fps, patterns)
        loss = clip_loss(fp_emb, patt_emb)
        
        optim.zero_grad()
        loss.backward()
        if config.get_param('max_grad_norm') > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.get_param('max_grad_norm'))
        optim.step()
        
        # compute validation loss
        with torch.no_grad():
            val_patterns, val_true_fps = train_dataset.get_validation(
                config.get_param('batch_size'))
            val_fp_emb, val_patt_emb = model(val_true_fps, val_patterns)
            val_loss = clip_loss(val_fp_emb, val_patt_emb)
        
        # log
        if config.get_param('log_every_steps'):
            if counter % config.get_param('log_every_steps') == 0:
                log_loss(args.loss_file, epoch, counter,
                         loss.item(), val_loss.item())
                print_update(epoch, batch_idx + 1,
                             loss.item(), val_loss.item())
        
        # early stopping
        early_stop(val_loss.item(), model, args.model_file, counter)
        if early_stop.stop:
            break
    
    if config.get_param('log_every_epochs'):
        log_loss(args.loss_file, epoch, counter,
                 loss.item(), val_loss.item())
        print_update(epoch, 'NA', loss.item(), val_loss.item())
    
    if early_stop.stop:
        break

# append information about the minimum validation loss
if config.get_param('log_every_epochs'):
    row = pd.DataFrame({'epoch': [None],
                        'minibatch': [early_stop.step_at_best],
                        'outcome': ['training loss'],
                        'value': [early_stop.best_loss]})
    row.to_csv(args.loss_file, index=False, mode='a', header=False)

predict_start_time = time.time()

# load the best model and make predictions on held-out peaks
model.load_state_dict(torch.load(args.model_file))
model.eval()

test_labels = pd.read_csv(args.test_mat_file, index_col=0)
test_cands = pd.read_csv(args.test_cands_file)
test_cands = test_cands.dropna(subset=['canonical_smiles', 'candidate_smiles'])

print(f'computing fingerprints for {len(test_cands)} test candidates...')
test_mols = clean_mols(test_cands['candidate_smiles'].values)
test_cand_fps = np.vstack([
    get_fingerprint(mol=mol,
                    fp_algo=config.get_param('fp_algo'),
                    fp_type=config.get_param('fp_type'),
                    length=config.get_param('fp_length'),
                    radius=config.get_param('fp_radius'))
    for mol in test_mols
])

results = []
unique_peak_ids = test_labels.index.unique()
with torch.no_grad():
    for peak_id in tqdm(unique_peak_ids, desc='predicting'):
        pattern = test_labels.loc[peak_id]
        if isinstance(pattern, pd.DataFrame):
            pattern = pattern.iloc[0]
        pattern = pattern.values.astype(np.float32)
        pattern_tensor = torch.from_numpy(pattern).to(device).unsqueeze(0)
        
        peak_cands = test_cands[test_cands['peak_id'] == peak_id]
        if len(peak_cands) == 0:
            continue
        
        cand_indices = peak_cands.index
        cand_fps = test_cand_fps[test_cands.index.get_indexer(cand_indices)]
        cand_fps_tensor = torch.from_numpy(
            cand_fps.astype(np.float32)).to(device)
        
        patt_emb = model.encode_patterns(pattern_tensor)
        cand_embs = model.encode_fingerprints(cand_fps_tensor)
        
        similarities = (cand_embs @ patt_emb.T).squeeze(-1).cpu().numpy()
        ranks = (-similarities).argsort().argsort() + 1
        
        for j, (idx, row) in enumerate(peak_cands.iterrows()):
            results.append({
                'peak_id': peak_id,
                'candidate_smiles': row['candidate_smiles'],
                'cosine': float(similarities[j]),
                'rank': int(ranks[j]),
                'match': row['match'] if 'match' in row else np.nan,
            })

results_df = pd.DataFrame(results).sort_values(['peak_id', 'rank'])
results_df.to_csv(args.pred_file, index=False)
print(f'wrote predictions for {len(results_df)} of {len(test_cands)} rows '
      f'to {args.pred_file}')

# write timing
load_time = train_start_time - start_time
train_time = predict_start_time - train_start_time
predict_time = time.time() - predict_start_time
total_time = time.time() - start_time
pd.DataFrame({
    'stage': ['load', 'train', 'predict', 'total'],
    'time': [load_time, train_time, predict_time, total_time],
}).to_csv(args.time_file, index=False)
