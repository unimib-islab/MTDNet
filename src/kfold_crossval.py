import argparse
import csv
import os
from datetime import datetime

import ipdb
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import SimpleDataset, SimpleGraphDataset
from model_RNN import *
from utils import seed_everything
from weight_initializers import init_weights
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser(description='K-Fold cross validation for preprocessed data. Subject-level splitting is performed',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-k', '--k', default=10,
                        type=int, help='number of folds')
    parser.add_argument('-n', '--ds_name', required=True,
                        help='name of the preprocessed dataset')
    parser.add_argument('-c', '--classes', required=True,
                        help='classes to use expressed as in annot file names (e.g. \'hc-ad\')')
    parser.add_argument('-p', '--ds_parent_dir', default='local/datasets/',
                        help='parent directory of the preprocessed dataset')
    parser.add_argument('-d', '--device', default='cuda:0',
                        help='device for computations (cuda:0, cpu, etc.)')
    parser.add_argument('-s', '--seed', default=1234,
                        type=int, help='seed to use')
    parser.add_argument('-b', '--batch_size', default=64,
                        type=int, help='training batch size')
    parser.add_argument('-w', '--num_workers', default=4,
                        type=int, help='number of workers for dataloaders')
    parser.add_argument('-e', '--num_epochs', default=100,
                        type=int, help='training epochs')
    parser.add_argument('-r', '--lr', default=0.00001,
                        type=float, help='training learning rate')
    parser.add_argument('-y', '--weight_decay', default=1e-8,
                        type=float, help='training weight decay')
    parser.add_argument('-g', '--scheduler_gamma', default=0.98,
                        type=float, help='exponential decay gamma')
    parser.add_argument('-l', '--loo', action='store_true',
                        help='ignore k and apply leave-one-out cross-validation (k = # samples)')
    parser.add_argument('--debug', action='store_true',
                        help='do not produce artifacts (no tensorboard logs, no saved checkpoints, etc)')
    parser.set_defaults(loo=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    return args


class CropPredCounter:
    """
    Predictions counter for 'crop' eval mode.
    """

    def __init__(self):
        self.gt_array = np.empty((0,), dtype=np.uint8)
        self.pred_array = np.empty((0,), dtype=np.uint8)

    def add_pred(self, gt, act):
        self.gt_array = np.append(self.gt_array, gt.astype(np.uint8))
        self.pred_array = np.append(
            self.pred_array, np.argmax(act).astype(np.uint8))

    def get_arrays(self):
        return self.gt_array, self.pred_array


class ConsensusPredCounter:
    """
    Predictions counter for 'consensus' eval mode.
    """

    def __init__(self):
        self.recs_gt = {}
        self.recs_preds = {}

    def add_pred(self, gt, act, rec_id):
        self.recs_gt[rec_id] = gt.astype(np.uint8)
        self.recs_preds[rec_id] = np.append(self.recs_preds.get(
            rec_id, np.empty((0,), dtype=np.uint8)), np.argmax(act).astype(np.uint8))

    def get_arrays(self):
        gt_array = np.empty((0,), dtype=np.uint8)
        pred_array = np.empty((0,), dtype=np.uint8)
        for i in self.recs_gt:
            gt_array = np.append(gt_array, self.recs_gt[i])
            pred_array = np.append(pred_array, np.argmax(
                np.bincount(self.recs_preds[i])).astype(np.uint8))
        return gt_array, pred_array


class AvgPredCounter:
    """
    Predictions counter for 'avg' eval mode.
    """

    def __init__(self):
        self.recs_gt = {}
        self.recs_preds = {}

    def add_pred(self, gt, act, rec_id):
        self.recs_gt[rec_id] = gt.astype(np.uint8)
        self.recs_preds[rec_id] = self.recs_preds.get(
            rec_id, np.zeros((act.shape[0],))) + softmax(act)

    def get_arrays(self):
        gt_array = np.empty((0,), dtype=np.uint8)
        pred_array = np.empty((0,), dtype=np.uint8)
        for i in self.recs_gt:
            gt_array = np.append(gt_array, self.recs_gt[i])
            # division not necessary, argmax of sum = argmax of avg
            pred_array = np.append(pred_array, np.argmax(
                self.recs_preds[i]).astype(np.uint8))
        return gt_array, pred_array


def compute_print_metrics(gt_array, pred_array, save=False, filename=None):
    acc = accuracy_score(gt_array, pred_array)
    class_report = classification_report(gt_array, pred_array)
    cm = confusion_matrix(gt_array, pred_array)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(class_report)
    print(cm)
    if save and filename is not None:
        with open(filename, 'a+') as f:
            print(f'\nAccuracy: {acc:.6f}\n', file=f)
            print(class_report, file=f)
            print(cm, file=f)


def train_one_epoch(model, epoch, tb_writer, loader, fold, device, optimizer, loss_fn):
    model.train()
    running_loss = 0.
    correct = 0

    for i, data in enumerate(tqdm(loader, ncols=100, desc='  Train')):

        # data = data.to(device)
        # inpt = data
        # label = data.y

        inpt = data[0].to(device)
        label = data[1].to(device).squeeze(1)
        optimizer.zero_grad()
        out = model(inpt)
        loss = loss_fn(out, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Running acc
        pred = out.argmax(dim=1)
        correct += int((pred == label).sum())

    print('  Logging to TensorBoard... ', end='')
    # log epoch loss
    avg_loss = running_loss / len(loader)
    tb_writer.add_scalar('Loss/train', avg_loss, epoch)
    # log epoch acc
    epoch_acc = correct / len(loader.dataset)
    tb_writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    print('done.')

    return (avg_loss, epoch_acc)


def val_one_epoch(model, epoch, tb_writer, loader, fold, device, loss_fn):
    model.eval()
    running_loss = 0.
    correct = 0

    for i, data in enumerate(tqdm(loader, ncols=100, desc='  Val')):
        # data = data.to(device)
        # inpt = data
        # label = data.y
        inpt = data[0].to(device)
        label = data[1].to(device).squeeze(1)
        out = model(inpt)
        loss = loss_fn(out, label)

        running_loss += loss.item()
        # Running acc
        pred = out.argmax(dim=1)
        correct += int((pred == label).sum())

    print('  Logging to TensorBoard... ', end='')
    # log epoch loss
    avg_loss = running_loss / len(loader)
    tb_writer.add_scalar('Loss/val', avg_loss, epoch)
    # log epoch acc
    epoch_acc = correct / len(loader.dataset)
    tb_writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    print('done.')

    return (avg_loss, epoch_acc)


def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_save_dir = f'local/checkpoints/train_{session_timestamp}/'
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    sumary_save_file = f'local/results/train_{session_timestamp}_{args.seed}.txt'
    os.makedirs('local/results/', exist_ok=True)

    crop_pred_counter = CropPredCounter()
    consensus_pred_counter = ConsensusPredCounter()
    avg_pred_counter = AvgPredCounter()

    annot_file_path = os.path.join(
        args.ds_parent_dir, args.ds_name, f"annot_all_{args.classes}.csv")
    crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name, "data")
    annotations = pd.read_csv(annot_file_path)
    subjects_list = annotations['original_rec'].unique().tolist()
    labels_list = [annotations[annotations['original_rec']
                               == s].iloc[0]['label'] for s in subjects_list]

    splitter = LeaveOneOut() if args.loo else StratifiedKFold(
        n_splits=args.k, random_state=69, shuffle=True)

    for fold, (train_idxs, val_idxs) in enumerate(splitter.split(np.zeros(len(labels_list)), labels_list)):
        #############################################
        # TODO experiments on reduced number of folds
        # if fold not in [0, 3, 6, 9]:
        #     continue
        #############################################
        seed_everything(args.seed)

        writer = SummaryWriter(
            'local/runs/train_{}_fold{}'.format(session_timestamp, str(fold)))

        train_subjects = [subjects_list[i] for i in train_idxs.tolist()]
        val_subjects = [subjects_list[i] for i in val_idxs.tolist()]
        train_df = annotations[annotations['original_rec'].isin(
            train_subjects)]  # crops in train set
        val_df = annotations[annotations['original_rec'].isin(
            val_subjects)]  # crops in val set

        train_dataset = SimpleDataset(train_df, crop_data_path)
        val_dataset = SimpleDataset(val_df, crop_data_path)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

        num_classes = args.classes.count('-') + 1
        model = RNN_EEG_v1(in_nch=19,
                           freq_bands=51,
                           pre_conv_weights=[256, 128, 64],
                           lstm_nch=512,
                           post_lin_weights=[128],
                           out_nch=num_classes)
        torch.save(model.state_dict(), './init_weights.pt')
        # init_weights(model, 'orthogonal')
        # model.load_state_dict(torch.load(os.path.join(
        #     './the_best_init_LSTM.pt'), map_location=device))
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma)

        print(
            f"\nStarting fold {fold}/{splitter.get_n_splits(np.zeros(len(labels_list)))}...")
        print(f'  Training samples: {len(train_dataset)}')
        print(f'  Validation samples: {len(val_dataset)}')

        # Training loop
        best_val_accuracy = 0
        for current_epoch in range(args.num_epochs):
            print(
                f'\n[Fold {fold}/{splitter.get_n_splits(np.zeros(len(labels_list)))}] Starting epoch {current_epoch:03d}.')
            train_loss, train_acc = train_one_epoch(
                model, current_epoch, writer, train_dataloader, fold, device, optimizer, loss_fn)
            
            scheduler.step()
            
            with torch.no_grad():
                val_loss, val_acc = val_one_epoch(
                    model, current_epoch, writer, val_dataloader, fold, device, loss_fn)

            writer.flush()
            print(f'[Fold {fold}/{splitter.get_n_splits(np.zeros(len(labels_list)))}] Epoch {current_epoch:03d} done.   Accuracy (tr/val): {train_acc:.4f}/{val_acc:.4f}   Loss (tr/val): {train_loss:.4f}/{val_loss:.4f}')

            # Save last model
            print("Saving checkpoint... ", end='')
            checkpoint_save_dict = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'epoch': current_epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint_save_dict, os.path.join(
                checkpoint_save_dir, f'fold_{fold}_last.pt'))
            print('done.')

            # Save best val acc model
            if val_acc > best_val_accuracy:
                print("New best val acc, saving checkpoint... ", end='')
                best_val_accuracy = val_acc
                torch.save(checkpoint_save_dict, os.path.join(
                    checkpoint_save_dir, f'fold_{fold}_best_val_acc.pt'))
                print('done.')

        # Eval
        print(
            f"\nEvaluating fold {fold}/{splitter.get_n_splits(np.zeros(len(labels_list)))}... ")
        model.load_state_dict(torch.load(os.path.join(
            checkpoint_save_dir, f'fold_{fold}_best_val_acc.pt'), map_location=device)['model_state_dict'])
        model.eval()
        for s in tqdm(range(len(val_df)), ncols=100):
            data = val_dataset[s]
            # data = data.to(device)
            # inpt = data
            # label = data.y
            inpt = data[0].to(device).unsqueeze(0)
            label = data[1].to(device)

            with torch.no_grad():
                out = model(inpt)

            crop_name = val_df.iloc[s]['crop_file']
            crop_gt = val_df.iloc[s]['label']
            crop_act = np.squeeze(out.detach().cpu().numpy())
            orig_rec = annotations[annotations['crop_file']
                                   == crop_name].iloc[0]['original_rec']
            crop_pred_counter.add_pred(crop_gt, crop_act)
            consensus_pred_counter.add_pred(crop_gt, crop_act, orig_rec)
            avg_pred_counter.add_pred(crop_gt, crop_act, orig_rec)

    # Final metrics
    # Crop pred counter
    print('\n  =======================> CROP <=======================')
    gt_array, pred_array = crop_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=sumary_save_file)
    # Consensus pred counter
    print('\n  =======================> CONSENSUS <=======================')
    gt_array, pred_array = consensus_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=sumary_save_file)
    # Avg pred counter
    print('\n  =======================> AVG <=======================')
    gt_array, pred_array = avg_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=sumary_save_file)


if __name__ == "__main__":
    main()
