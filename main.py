#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#

import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from bioFAME.utils import compute_acc, collect_cm, set_random_seed, linear_clf_train

from bioFAME.datasets.dataloader import bioFAME_data
from bioFAME.models.algorithms import get_algorithm_class
from bioFAME.models.hparams_registry import _hparams
from bioFAME.transforms.augmentation import Compose, Downsample
from bioFAME.transforms.mixup_helper import Mixup_helper

import logging
logger = logging.getLogger("bioFAME")
logger.setLevel(logging.DEBUG)


def setup_parser(parser: argparse.ArgumentParser) -> None:

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_name', type=str, default='SleepEDF_multi')
    parser.add_argument('--train_data_file', type=str, default='train.pt')
    parser.add_argument('--val_data_file', type=str, default='val.pt')
    parser.add_argument('--test_data_file', type=str, default='test.pt')
    parser.add_argument('--train_length', type=int, default=3000)
    parser.add_argument('--test_length', type=int, default=3000)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu', choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--algorithm', type=str, default='bioFAME')
    parser.add_argument("--skip_validation", action='store_true')

    # logger information
    parser.add_argument('--tb_dir', type=str, default='./TB_logs')
    parser.add_argument('--tb_name', type=str, default='visualization')

    # dataset settings
    parser.add_argument('--modality', type=list, default=['eeg']) # 
    parser.add_argument('--mod_channels', type=int, default=1)

    # training settings
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--reconstruction", action='store_true')

    # evaluation settings
    parser.add_argument('--test_every_epoch', type=int, default=3)

    # fine-tune settings
    parser.add_argument("--fine_tune", action='store_true', help='Fine tuning flag')
    parser.add_argument('--ft_load_model', action='store_true', 
        help='If exists it should load model ckpt. If no ckpt existing, use random init (forced-benchmark)')

    parser.add_argument("--ft_new_modality", action='store_true')
    parser.add_argument("--ft_new_data", action='store_true')

    parser.add_argument('--ft_data_dir', type=str)
    parser.add_argument('--ft_data_name', type=str, default='FD_B')
    parser.add_argument('--ft_train_data_file', type=str, default='train.pt')
    parser.add_argument('--ft_val_data_file', type=str, default='val.pt')
    parser.add_argument('--ft_test_data_file', type=str, default='test.pt')

    parser.add_argument('--ft_classes', type=int, default=3)
    parser.add_argument("--ft_channel", type=list, default=['default'])
    parser.add_argument('--ft_n_channel', type=list, default=[1])
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument("--ft_lr", default=0.001, type=float)


def setup_dataloader(
    args, 
    data_path, 
    train_file, 
    val_file, 
    test_file, 
    modality=None,
    transforms=None,
    dataset_name=None,
    ):
    logger.info("Setting up dataloader transforms for {} on {}...".format(dataset_name, modality))

    if modality is None:
        modality = ['default']

    train_dataset = bioFAME_data(data_path, filename=train_file, channels=modality, transforms=transforms[0], dataset_name=dataset_name)
    val_dataset = bioFAME_data(data_path, filename=val_file, channels=modality, transforms=transforms[1], dataset_name=dataset_name)
    test_dataset = bioFAME_data(data_path, filename=test_file, channels=modality, transforms=transforms[2], dataset_name=dataset_name)

    logger.info("  Num examples:")
    logger.info(f"    Train: {len(train_dataset)}")
    logger.info(f"    Val: {len(val_dataset)}")
    logger.info(f"    Test:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers != 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers != 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers != 0),
    )

    return train_loader, val_loader, test_loader


class Trainer():
    def __init__(self, args, train_loader, val_loader, test_loader, model_hparams) -> None:
        super(Trainer, self).__init__()

        self.args = args
        self.device = args.device
        self.train_loader, self.test_loader, self.val_loader = train_loader, test_loader, val_loader

        # setup tb_logger and device
        tb_position = args.tb_dir
        self.exp_name = f"{args.tb_name}_{args.algorithm}"
        self.tb_logger = TensorBoardLogger(tb_position, name=self.exp_name)
        self.device = torch.device(args.device)

        # model-related
        self.skip_val = args.skip_validation
        self.lr = args.lr

        model_class = get_algorithm_class(args.algorithm)
        self.model = model_class(in_channel=args.mod_channels, length=args.train_length, n_classes=args.classes, hparams=model_hparams)
        self.model.to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), args.lr, weight_decay=0.0001)
        self.global_step = 0

        self.mixup_helper = Mixup_helper(mode=model_hparams['mixup_mode'])

    def train_epoch(self, recon=False):
        self.model.train()

        for i, (data, label) in enumerate(tqdm(self.train_loader)):
            self.global_step += 1
            data, label = data.to(self.device), label.long().to(self.device)

            if recon:
                loss = self.model(data)

            else:
                if self.mixup_helper is None:
                    pred = self.model(data)
                    loss = F.cross_entropy(pred, label, weight=self.weight)

                else:
                    mixed_data, mixed_labels_content = self.mixup_helper.forward((data, label))
                    pred = self.model(mixed_data)

                    if type(mixed_labels_content) == type((0, 1, 2)):
                        lam, yi, yj = mixed_labels_content
                        loss = lam * F.cross_entropy(pred, yi)
                        loss += (1 - lam) * F.cross_entropy(pred, yj)
                        
                    else:
                        loss = F.cross_entropy(pred, mixed_labels_content)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.tb_logger.log_metrics({'Loss/train_loss': loss.item(),}, step=self.global_step)

    def eval_epoch(self, recon=False):
        collect_results = {}
        self.model.eval()

        if recon:
            clf_class = get_algorithm_class('SSL_MLP')

            classifier = clf_class(latent_dim=self.model.linear_clf_dim, n_classes=self.args.classes).to(self.device)
            clf_optim = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)
            clf_trainer = linear_clf_train(
                self.model, 
                classifier, 
                clf_optim, 
                self.train_loader, 
                self.test_loader, 
                device = self.device,
                num_epochs=50,
                val_dataloader= None if self.skip_val else self.val_loader, 
                )

            train_acc, test_acc = clf_trainer.train_acc, clf_trainer.test_acc

            collect_results['train_acc'] = train_acc
            collect_results['test_acc'] = test_acc

        else:
            train_acc = compute_acc(self.model, self.train_loader, self.device, None)
            if not self.skip_val:
                val_acc = compute_acc(self.model, self.val_loader, self.device, None)
                collect_results['val_acc'] = val_acc
            test_acc = compute_acc(self.model, self.test_loader, self.device, None)
            
            collect_results['train_acc'] = train_acc
            collect_results['test_acc'] = test_acc

            precision, recall, F1, cm, bal_binary = collect_cm(self.model, self.test_loader, self.device, None)

            collect_results['precision'] = precision
            collect_results['recall'] = recall
            collect_results['F1'] = F1
            collect_results['bal_binary'] = bal_binary

        return collect_results

    def main(self, recon=False, TB_name='TrainAcc', tt_epoch=50):
        logger.info('   Start model training! train_recon {}'.format(recon))

        # normal model training and evaluation
        for epoch in range(tt_epoch):
            self.train_epoch(recon=recon)

            if epoch % args.test_every_epoch == 0:
                if recon:
                    self.model._mode_adjustment(encoder_only=True)
                    results = self.eval_epoch(recon=recon)
                    self.model._mode_adjustment(encoder_only=False)
                else:
                    results = self.eval_epoch(recon=recon)

                self._dict_to_logs(results, TB_name, epoch)

    def _dict_to_logs(self, results_dict, TB_name, epoch):
        tb_logger_dict = {}
        for key in results_dict:
            tb_logger_dict['{}/{}'.format(TB_name, key)] = results_dict[key]
        self.tb_logger.log_metrics(tb_logger_dict, step=epoch)
        
    def _update_lr(self, new_lr):
        '''update learning rate for e.g. fine-tuning.'''
        for g in self.optim.param_groups:
                g['lr'] = new_lr # args.ft_lr

    def _update_clf(self, n_classes, new_lr=None, new_channel=None):
        if new_channel is not None:
            self.model._channel_adjustment(new_channel=new_channel)
            self.model._renew_revin(new_channel=new_channel)
        self.model._clf_refresh(n_classes=n_classes)
        self.model.to(self.device)
        if new_lr is None:
            new_lr = self.lr
        self.optim = torch.optim.Adam(self.model.parameters(), new_lr, weight_decay=0.0001)

    def _new_data(self, train_loader, val_loader, test_loader):
        '''re-write dataloader'''
        self.train_loader, self.test_loader = train_loader, test_loader
        if val_loader is not None:
            self.val_loader = val_loader

    def _save_model(self, save_path='./model_ckpt'):
        model_save_path = os.path.join(save_path, self.exp_name, 'ckpt.pt')
        if not os.path.exists(os.path.join(save_path, self.exp_name)):
            os.makedirs(os.path.join(save_path, self.exp_name))
        torch.save(self.model.state_dict(), model_save_path)
    
    def _load_model(self, load_path='./model_ckpt'):
        logger.info('******** You have asked bioFAME to load your model checkpoint')
        model_load_path = os.path.join(load_path, self.exp_name, 'ckpt.pt')
        if not os.path.exists(os.path.join(load_path, self.exp_name)):
            logger.info('   I cannot find the ckpt! Start from random init instead!')
            return
        else:
            logger.info('   We have found your ckpt path')

        model_ckpt = torch.load(model_load_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_ckpt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='bioFAME training pipeline')
    setup_parser(parser)
    args = parser.parse_args()

    set_random_seed(args.seed)
    default_aug = Compose([Downsample(new_size=args.train_length)])
    train_aug = Compose([Downsample(new_size=args.train_length)])

    default_aug_test = Compose([Downsample(new_size=args.test_length)])
    train_aug_test = Compose([Downsample(new_size=args.test_length)])

    train_loader, val_loader, test_loader = setup_dataloader(
        args,
        data_path=args.data_dir, 
        train_file=args.train_data_file,
        val_file=args.val_data_file,
        test_file=args.test_data_file,
        modality=args.modality,
        transforms=[default_aug, default_aug, default_aug],
        dataset_name=args.data_name,
        )

    exp_data, exp_label = next(iter(train_loader))
    logger.info("   Example data check")
    logger.info("   Data shape {}, label shape {}".format(exp_data.shape, exp_label.shape))

    model_hparams = _hparams(args.algorithm)
    trainer = Trainer(args, train_loader, val_loader, test_loader, model_hparams)

    if not args.ft_load_model:
        trainer.main(
            recon=args.reconstruction,
            TB_name='TrainAcc',
            tt_epoch=args.epochs)

        trainer._save_model()
    else:
        trainer._load_model()
    
    # enter fine-tuning stage
    if args.fine_tune:

        if args.ft_new_modality:

            for i, new_modality in enumerate(args.ft_channel):
                logger.info('Replacing FT data with new channel {}'.format(new_modality))

                ft_train_loader, ft_val_loader, ft_test_loader = setup_dataloader(
                    args, 
                    data_path=args.data_dir, 
                    train_file=args.train_data_file, 
                    val_file=args.val_data_file,
                    test_file=args.test_data_file,
                    modality=[new_modality],
                    transforms=[train_aug_test, default_aug_test, default_aug_test],
                    dataset_name=args.data_name,
                    )

                trainer._new_data(ft_train_loader, ft_val_loader, ft_test_loader)

                if len(args.ft_n_channel) == 1:
                    ft_n_channel_feed = args.ft_n_channel[0]
                else:
                    assert len(args.ft_n_channel) == len(args.ft_channel)
                    ft_n_channel_feed = args.ft_n_channel[i]

                trainer.model._channel_adjustment(new_channel=ft_n_channel_feed)
                trainer._update_clf(n_classes=args.ft_classes, new_lr=args.ft_lr, new_channel=ft_n_channel_feed)

                trainer.model._mode_adjustment(encoder_only=True)
                trainer.main(train_recon=False, eval_SSL=False, TB_name='FT_{}'.format(new_modality), tt_epoch=args.ft_epochs)
        
        if args.ft_new_data:
            logger.info('Replacing FT data with a new dataset')
            
            ft_train_loader, ft_val_loader, ft_test_loader = setup_dataloader(
                args, 
                data_path=args.ft_data_dir, 
                train_file=args.ft_train_data_file, 
                val_file=args.ft_val_data_file,
                test_file=args.ft_test_data_file,
                modality=args.ft_channel,
                transforms=[train_aug_test, default_aug_test, default_aug_test],
                dataset_name=args.ft_data_name,
                )

            ft_exp_data, ft_exp_label = next(iter(ft_train_loader))
            logger.info("   Example data check during fine-tuning")
            logger.info("   Data shape {}, label shape {}".format(ft_exp_data.shape, ft_exp_label.shape))

            trainer._new_data(ft_train_loader, ft_val_loader, ft_test_loader)

            trainer._update_clf(
                n_classes=args.ft_classes, new_lr=args.ft_lr, new_channel=args.ft_n_channel[0],)

            trainer.model._mode_adjustment(encoder_only=True)
            trainer.main(recon=False, TB_name='FT', tt_epoch=args.ft_epochs)

