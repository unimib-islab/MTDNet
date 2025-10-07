import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import SimpleDataset
from model import RNN_EEG
from utils import count_parameters, seed_everything, tb_option_parser


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
    parser.add_argument('-ld', '--log_dir', default='local/runs/',
                        help='directory for tensorboard logs')
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
    parser.add_argument('-ch', '--el_channels', default=19,
                        type=int, help='number of electrodes')
    parser.add_argument('-r', '--lr', default=0.00001,
                        type=float, help='training learning rate')
    parser.add_argument('-y', '--weight_decay', default=1e-8,
                        type=float, help='training weight decay')
    parser.add_argument('-g', '--scheduler_gamma', default=0.98,
                        type=float, help='exponential decay gamma')
    parser.add_argument('-l', '--loo', action='store_true',
                        help='ignore k and apply leave-one-out cross-validation (k = # samples)')
    parser.add_argument('-lf', '--log_file', default='local/runs/log.txt',
                        help='file to log accuracy and f1-score')
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
    
    return acc, f1


def train_one_epoch(model, epoch, tb_writer, loader, fold, device, optimizer, loss_fn):
    model.train()
    running_loss = 0.
    correct = 0

    for i, data in enumerate(tqdm(loader, ncols=100, desc='  Train')):

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

def test_one_epoch(model, epoch, tb_writer, loader, fold, device, loss_fn):
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

    # import ipdb; ipdb.set_trace()
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
    save_log_dir = args.log_dir
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_save_dir = f'local/checkpoints/train_{session_timestamp}/'
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    summary_save_file = f'local/results/train_{session_timestamp}_{args.seed}.txt'
    os.makedirs('local/results/', exist_ok=True)

    crop_pred_counter = CropPredCounter()
    consensus_pred_counter = ConsensusPredCounter()
    avg_pred_counter = AvgPredCounter()

    annot_file_path = os.path.join(
        args.ds_parent_dir, args.ds_name, f"annot_all_{args.classes}.csv")
    crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name)
    annotations = pd.read_csv(annot_file_path)
    subjects_list = annotations['original_rec'].unique().tolist()
    labels_list = [annotations[annotations['original_rec']
                               == s].iloc[0]['label'] for s in subjects_list]

    splitter = LeaveOneOut() if args.loo else StratifiedKFold(
        n_splits=args.k, random_state=69, shuffle=True)



    seed_everything(args.seed)

    writer = SummaryWriter(
        save_log_dir+'/train_{}'.format(session_timestamp))
    
    if args.classes == "hc-ad":
        #ADFTD
        if "miltiadous" in args.ds_name:
            train_subjects = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            val_subjects = [54, 55, 56, 57, 58, 59, 22, 23, 24, 25, 26, 27, 28]
            test_subjects = [60, 61, 62, 63, 64, 65, 29, 30, 31, 32, 33, 34, 35, 36]
            train_subjects = [ 'sub-{:03d}'.format(s) for s in train_subjects ]    
            val_subjects  = [ 'sub-{:03d}'.format(s) for s in val_subjects ]
            test_subjects = [ 'sub-{:03d}'.format(s) for s in test_subjects ]
        #ADSZ
        elif "ADSZ" in args.ds_name:
            train_subjects = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            val_subjects = [39, 40, 41, 42, 43, 15, 16, 17, 18, 19]
            test_subjects = [44, 45, 46, 47, 48, 20, 21, 22, 23, 24]
            train_subjects = [ 'sub-{:02d}'.format(s) for s in train_subjects ]    
            val_subjects  = [ 'sub-{:02d}'.format(s) for s in val_subjects ]
            test_subjects = [ 'sub-{:02d}'.format(s) for s in test_subjects ]
        #APAVA
        elif "APAVA" in args.ds_name:
            train_subjects = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23]
            val_subjects = [15, 16, 19, 20]
            test_subjects = [1, 2, 17, 18]
            train_subjects = [ 'sub-{:02d}'.format(s) for s in train_subjects ]    
            val_subjects  = [ 'sub-{:02d}'.format(s) for s in val_subjects ]
            test_subjects = [ 'sub-{:02d}'.format(s) for s in test_subjects ]
        elif "brainlat" in args.ds_name:
            train_subjects = [ "1_AD_AR_sub-30020",
                                "1_AD_CL_sub-30007",
                                "1_AD_AR_sub-30008",
                                "1_AD_AR_sub-30001",
                                "1_AD_AR_sub-30015",
                                "1_AD_CL_sub-30025",
                                "1_AD_AR_sub-30002",
                                "1_AD_CL_sub-30034",
                                "1_AD_CL_sub-30010",
                                "1_AD_AR_sub-30013",
                                "1_AD_AR_sub-30012",
                                "1_AD_AR_sub-30031",
                                "1_AD_AR_sub-30026",
                                "1_AD_AR_sub-30029",
                                "1_AD_AR_sub-30022",
                                "1_AD_CL_sub-30003",
                                "1_AD_CL_sub-30014",
                                "1_AD_CL_sub-30023",
                                "1_AD_CL_sub-30032",
                                "1_AD_CL_sub-30006",
                                "1_AD_CL_sub-30021",
                                "5_HC_AR_sub-10004",
                                "5_HC_AR_sub-10009",
                                "5_HC_AR_sub-100035",
                                "5_HC_AR_sub-100015",
                                "5_HC_CL_sub-100016",
                                "5_HC_AR_sub-100028",
                                "5_HC_AR_sub-10003",
                                "5_HC_CL_sub-100017",
                                "5_HC_AR_sub-100022",
                                "5_HC_AR_sub-10007",
                                "5_HC_AR_sub-100033",
                                "5_HC_CL_sub-100010",
                                "5_HC_AR_sub-100038",
                                "5_HC_AR_sub-100026",
                                "5_HC_CL_sub-100021",
                                "5_HC_CL_sub-10001",
                                "5_HC_CL_sub-10008",
                                "5_HC_CL_sub-100043",
                                "5_HC_AR_sub-100012",
                                "5_HC_AR_sub-100030" ]    
            val_subjects  = [ "1_AD_AR_sub-30009",
                                "1_AD_CL_sub-30028",
                                "1_AD_CL_sub-30033",
                                "1_AD_CL_sub-30016",
                                "1_AD_AR_sub-30018",
                                "1_AD_CL_sub-30024",
                                "1_AD_AR_sub-30011",
                                "5_HC_CL_sub-100014",
                                "5_HC_AR_sub-100024",
                                "5_HC_AR_sub-100020",
                                "5_HC_AR_sub-10006",
                                "5_HC_CL_sub-100037",
                                "5_HC_CL_sub-100034"]
            test_subjects = [ "1_AD_CL_sub-30005",
                                "1_AD_CL_sub-30030",
                                "1_AD_AR_sub-30004",
                                "1_AD_CL_sub-30035",
                                "1_AD_CL_sub-30019",
                                "1_AD_AR_sub-30017",
                                "1_AD_CL_sub-30027",
                                "5_HC_CL_sub-10005",
                                "5_HC_AR_sub-100018",
                                "5_HC_CL_sub-100011",
                                "5_HC_AR_sub-10002",
                                "5_HC_CL_sub-100029",
                                "5_HC_AR_sub-100031" ]
        else:
            raise Exception("wrong db split") 
        
        
    elif args.classes == "hc-ftd-ad":
        train_subjects = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        val_subjects = [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28]
        test_subjects = [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
        train_subjects = [ 'sub-{:03d}'.format(s) for s in train_subjects ]    
        val_subjects  = [ 'sub-{:03d}'.format(s) for s in val_subjects ]
        test_subjects = [ 'sub-{:03d}'.format(s) for s in test_subjects ]




    train_df = annotations[annotations['original_rec'].isin(
        train_subjects)]  # crops in train set
    val_df = annotations[annotations['original_rec'].isin(
        val_subjects)]  # crops in val set

    train_dataset = SimpleDataset(train_df, crop_data_path)
    val_dataset = SimpleDataset(val_df, crop_data_path, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)


    test_df = annotations[annotations['original_rec'].isin(
        test_subjects)]  # crops in val set

    test_dataset = SimpleDataset(test_df, crop_data_path, mode='test')

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)



    num_classes = args.classes.count('-') + 1

    model = RNN_EEG(in_nch=train_dataset.get_electrode_number(),
                       first_layer_ch = 32*2,
                       lstm_nch=16,
                       post_lin_weights=[16],
                       out_nch=num_classes)


    infos = {}
    
    infos['n_params'] = count_parameters(model)
    infos['n_classes'] = num_classes
    infos['learning_rate'] = args.lr
    tb_option_parser(infos, writer)

    model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.scheduler_gamma)

    print(f'  Training samples: {len(train_dataset)}')
    print(f'  Validation samples: {len(val_dataset)}')

    # Training loop
    best_val_accuracy = 0
    for current_epoch in range(args.num_epochs):
        print(
            f'\nStarting epoch {current_epoch:03d}.')
        train_loss, train_acc = train_one_epoch(
            model, current_epoch, writer, train_dataloader, 0, device, optimizer, loss_fn)
        
        scheduler.step()
        
        with torch.no_grad():
            val_loss, val_acc = val_one_epoch(
                model, current_epoch, writer, val_dataloader, 0, device, loss_fn)
            
            test_loss, test_acc = test_one_epoch(
                model, current_epoch, writer, test_dataloader, 0, device, loss_fn)

        writer.flush()
        print(f'Epoch {current_epoch:03d} done.   Accuracy (tr/val/test): {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}   Loss (tr/val/test): {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}')

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
            checkpoint_save_dir, f'_last.pt'))
        print('done.')

        # Save best val acc model
        if val_acc > best_val_accuracy:
            print("New best val acc, saving checkpoint... ", end='')
            best_val_accuracy = val_acc
            torch.save(checkpoint_save_dict, os.path.join(
                checkpoint_save_dir, f'_best_val_acc.pt'))
            print('done.')

    # Eval



    print(
        f"\nEvaluating ... ")
    model.load_state_dict(torch.load(os.path.join(
        checkpoint_save_dir, f'_last.pt'), map_location=device)['model_state_dict'])
    model.eval()
    for s in tqdm(range(len(test_df)), ncols=100):
        data = test_dataset[s]

        inpt = data[0].to(device).unsqueeze(0)
        label = data[1].to(device)

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

    # Final metrics
    # Crop pred counter
    print('\n  =======================> CROP <=======================')
    gt_array, pred_array = crop_pred_counter.get_arrays()
    crop_acc, crop_f1 = compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)
    # Consensus pred counter
    print('\n  =======================> CONSENSUS <=======================')
    gt_array, pred_array = consensus_pred_counter.get_arrays()
    cons_acc, cons_f1 = compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)
    # Avg pred counter
    print('\n  =======================> AVG <=======================')
    gt_array, pred_array = avg_pred_counter.get_arrays()
    avg_acc, avg_f1 = compute_print_metrics(gt_array, pred_array, save=True,
                          filename=summary_save_file)

    with open(args.log_file, "a") as myfile:
        myfile.write(f"\n{crop_acc}, {crop_f1}, {cons_acc}, {cons_f1}, {avg_acc}, {avg_f1}")



if __name__ == "__main__":
    main()
