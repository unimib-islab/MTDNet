import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

from datasets import SimpleDataset
from model import RNN_EEG
from utils import seed_everything


def argparser():
    parser = argparse.ArgumentParser(description='Model testing on ADFormer split',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--ds_parent_dir', default='local/datasets/',
                        help='parent directory of the preprocessed dataset')

    parser.add_argument('-n', '--ds_name', required=True,
                        help='name of the preprocessed dataset')
    parser.add_argument('-d', '--device', default='cuda:0',
                        help='device for computations (cuda:0, cpu, etc.)')
    parser.add_argument('-s', '--seed', default=1234,
                        type=int, help='seed to use')

    parser.add_argument('-w', '--num_workers', default=4,
                        type=int, help='number of workers for dataloaders')
    parser.add_argument('-c', '--classes', required=True,
                        help='classes to use expressed as in annot file names (e.g. \'hc-ad\')')

    parser.add_argument('-ckp', '--checkpoint', default=None,
                        type=str, help='directory of model to load')
    
    parser.add_argument('-v', '--version', default='best',
                        help='checkpoint version (best or last)')

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
    f1 = f1_score(gt_array, pred_array, average='macro')
    class_report = classification_report(gt_array, pred_array)
    cm = confusion_matrix(gt_array, pred_array)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(f'F1-Score: {f1:.6f}\n')
    print(class_report)
    print(cm)
    if save and filename is not None:
        with open(filename, 'a+') as f:
            print(f'\nAccuracy: {acc:.6f}\n', file=f)
            print(f'F1-Score: {f1:.6f}\n', file=f)
            print(class_report, file=f)
            print(cm, file=f)


def test_one_epoch(model, epoch, tb_writer, loader, fold, device, loss_fn):
    model.eval()
    running_loss = 0.
    correct = 0

    for i, data in enumerate(tqdm(loader, ncols=100, desc='  Val')):

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
    tb_writer.add_scalar('Loss/test', avg_loss, epoch)
    # log epoch acc
    epoch_acc = correct / len(loader.dataset)
    tb_writer.add_scalar('Accuracy/test', epoch_acc, epoch)
    print('done.')

    return (avg_loss, epoch_acc)


def main():

    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_to_load = args.checkpoint

    crop_pred_counter = CropPredCounter()
    consensus_pred_counter = ConsensusPredCounter()
    avg_pred_counter = AvgPredCounter()

    annot_file_path = os.path.join(
        args.ds_parent_dir, args.ds_name, f"annot_all_{args.classes}.csv")
    crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name)
    annotations = pd.read_csv(annot_file_path)

    seed_everything(args.seed)

    model_name = args.checkpoint.split('/')[-2]

    summary_save_file = f'local/results_paper/{model_name}.txt'

    save_feat_path = summary_save_file[:-4]+f'_features/'
    
    if not os.path.exists(save_feat_path):
        os.makedirs(save_feat_path)

    if args.classes == "hc-ad":
        #ADFTD
        if "miltiadous" in args.ds_name:
            test_subjects = [60, 61, 62, 63, 64, 65, 29, 30, 31, 32, 33, 34, 35, 36]
            test_subjects = [ 'sub-{:03d}'.format(s) for s in test_subjects ]
        #ADSZ
        elif "ADSZ" in args.ds_name:

            test_subjects = [44, 45, 46, 47, 48, 20, 21, 22, 23, 24]
            test_subjects = [ 'sub-{:02d}'.format(s) for s in test_subjects ]
        #APAVA
        elif "APAVA" in args.ds_name:
            test_subjects = [1, 2, 17, 18]
            test_subjects = [ 'sub-{:02d}'.format(s) for s in test_subjects ]
        else:
            raise Exception("wrong db split") 
        
        
    elif args.classes == "hc-ftd-ad":
        test_subjects = [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
        test_subjects = [ 'sub-{:03d}'.format(s) for s in test_subjects ]


    test_df = annotations[annotations['original_rec'].isin(
        test_subjects)]  # crops in val set

    test_dataset = SimpleDataset(test_df, crop_data_path, mode='test')


    num_classes = args.classes.count('-') + 1
     

    model = RNN_EEG(in_nch=test_dataset.get_electrode_number(),
                       first_layer_ch = 32,
                       lstm_nch=16,
                       post_lin_weights=[16],
                       out_nch=num_classes)

    model.to(device)

    print(
        f"\nEvaluating ... ")
    if args.version == 'last':
        model.load_state_dict(torch.load(os.path.join(
            model_to_load, f'_last.pt'), map_location=device)['model_state_dict'])
    elif args.version == 'best':
        model.load_state_dict(torch.load(os.path.join(
            model_to_load, f'_best_val_acc.pt'), map_location=device)['model_state_dict'])
    else:
        raise Exception("Unknown checkpoint version")
    
    model.eval()
    idx = 0
    for s in tqdm(range(len(test_df)), ncols=100):
        data = test_dataset[s]

        inpt = data[0].to(device).unsqueeze(0)
        label = data[1].to(device)
        subject = data[2]

        with torch.no_grad():
            out = model(inpt)

        crop_name = test_df.iloc[s]['crop_file']
        crop_gt = test_df.iloc[s]['label']
        crop_act = np.squeeze(out.detach().cpu().numpy())
        orig_rec = annotations[annotations['crop_file']
                                == crop_name].iloc[0]['original_rec']
        
        crop_pred_counter.add_pred(crop_gt, crop_act)
        consensus_pred_counter.add_pred(crop_gt, crop_act, orig_rec)
        avg_pred_counter.add_pred(crop_gt, crop_act, orig_rec)

        # save features when label is correct: TODO
        # if label == torch.argmax(out):
        feat = {}
        feat['sf1'] = model.last_s1f.cpu().detach().numpy()
        feat['sf2'] = model.last_s2f.cpu().detach().numpy()
        feat['sf3'] = model.last_s3f.cpu().detach().numpy()
        feat['gt label'] = label.cpu().numpy()
        feat['out label'] = torch.argmax(out).item()
        feat['subject'] = subject

        with open(save_feat_path+f'/feat_{idx:05d}.pkl', 'wb') as ff:
            pickle.dump(feat, ff)
        idx+=1
    # Final metrics
    # Crop pred counter
    print('\n  =======================> CROP <=======================')
    gt_array, pred_array = crop_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)
    # Consensus pred counter
    print('\n  =======================> CONSENSUS <=======================')
    gt_array, pred_array = consensus_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)
    # Avg pred counter
    print('\n  =======================> AVG <=======================')
    gt_array, pred_array = avg_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)

    
       


if __name__ == "__main__":
    main()
