import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from our_code.dataloader import CTDoseDataLoader as dataloader
from our_code.model_pix2pix import init_weights, UnetMaxGenerator
import torch
from provided_code.general_functions import get_paths, make_directory_and_return_path
import numpy as np
import shutil
from our_code._3D_loss import _3DLoss
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from argparse import ArgumentParser
from provided_code.general_functions import sparse_vector_function
import pandas as pd
import torchvision



np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ganloss(torch.nn.Module):
    def __init__(self):
        super(ganloss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, pred, value):

        target = torch.tensor(value).expand_as(pred).cuda()
        return self.loss(pred, target)


class DosePrediction(pl.LightningModule):
    def __init__(self, hparams):
        super(DosePrediction, self).__init__()

        self.hparams = hparams
        self.model = UnetMaxGenerator(11, 1, self.hparams.down_samples, ngf=self.hparams.ngf, norm_layer=self.hparams.norm, use_resnet=self.hparams.use_resnet, used_act=self.hparams.used_act)
        init_weights(self.model, init_type='normal')


        self.loss = _3DLoss(torch.device('cuda:0'), num_layers=self.hparams.num_layers, weights=self.hparams.loss_weights, use_mask=self.hparams.use_mask)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.plan_paths = get_paths(self.hparams.training_data_dir, ext='')
        self.num_train_pats = np.minimum(self.hparams.num_train_pats, len(self.plan_paths))  # number of plans that will be used to train model
        self.validation_data_paths = get_paths(self.hparams.validation_data_dir, ext='')


    def train_dataloader(self):
        training_paths = self.plan_paths[:self.hparams.num_train_pats]  # list of training plans
        self.data_loader_train = dataloader(training_paths, batch_size=self.hparams.batchsize, trans=['flip', 128, 'crop'])
        return self.data_loader_train

    def val_dataloader(self):
        hold_out_paths = self.plan_paths[self.hparams.num_train_pats:]  # list of paths used for held out testing
        self.data_loader_eval = dataloader(hold_out_paths, batch_size=1, trans=[128])
        return self.data_loader_eval

    def test_dataloader(self):
        self.data_loader_test = dataloader(self.validation_data_paths, batch_size=1, trans=[128],
                                           mode_name='dose_prediction')
        return self.data_loader_test

    def configure_optimizers(self):
        optim_G = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999), weight_decay=self.hparams.wd)

        schedular_G = {'scheduler':torch.optim.lr_scheduler.OneCycleLR(optim_G, max_lr=self.hparams.lr,
                                                             steps_per_epoch=int(np.ceil(self.num_train_pats/self.hparams.batchsize)),
                                                             epochs=self.hparams.epochs),
                      'interval': 'step'}

        return [optim_G], [schedular_G]

    def training_step(self, batch, batch_idx):
        ct, dose, mask, structure = batch['ct'], batch['dose'], batch['possible_dose_mask'], batch['structure_masks']

        input = torch.cat([ct, structure], dim=1)
        pred = self(input)*mask
        loss = self.loss(pred, dose, mask)

        with torch.no_grad():
            dose_score = (torch.sum(torch.abs((dose*100)-(pred*100)))/torch.sum(mask))


        if batch['patient_list'][0][0] == 'pt_40':

            images = [dose.data[:, 0:1, 64, :, :], dose.data[:, 0:1, :, 64, :], dose.data[:, 0:1, :, :, 64],
                      pred.data[:, 0:1, 64, :, :], pred.data[:, 0:1, :, 64, :], pred.data[:, 0:1, :, :, 64],
                      dose.data[:, 0:1, 64, :, :]- pred.data[:, 0:1, 64, :, :], dose.data[:, 0:1, :, 64, :]-pred.data[:, 0:1, :, 64, :],
                      dose.data[:, 0:1, :, :, 64] - pred.data[:, 0:1, :, :, 64]
                      ]
            image_stack = torchvision.utils.make_grid(torch.cat(images, dim=0), nrow=3)
            self.logger.experiment.log({'generated images': wandb.Image(image_stack)})

        log_dict = {'train_dose_score': dose_score}

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}


    def training_epoch_end(self, outputs):

        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        log_dict = {'loss_mean': loss_mean}

        return {'loss_mean': loss_mean, 'log': log_dict, 'progress_bar': log_dict}


    def validation_step(self, batch, batch_idx):
        ct, dose, mask, structure = batch['ct'].float(), batch['dose'].float(), batch['possible_dose_mask'].float(), batch['structure_masks'].float()

        pred = self(torch.cat([ct, structure], dim=1)) * mask

        dose_score = (torch.sum(torch.abs((dose * 100) - (pred * 100))) / torch.sum(mask))

        #TODO: include addtional metrics for evaluation--> use already coded version in template
        #DVH_loss = DVH(pred, dose, structure)

        log_dict = {'val_loss': dose_score, 'train_dose_score': dose_score}

        return {'val_loss': dose_score, 'log': log_dict, 'progress_bar': log_dict}


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss_mean': val_loss_mean}
        return {'val_loss_mean': val_loss_mean, 'log': log_dict, 'progress_bar': log_dict}

    def test_step(self, batch, batch_idx):
        ct, mask, structure, pat_id = batch['ct'], batch['possible_dose_mask'], batch['structure_masks'], batch['patient_list'][0][0]

        pred = self(torch.cat([ct, structure], dim=1).float()) * mask * 100

        dose_pred_gy = np.transpose(pred.squeeze(0).squeeze(0).cpu().numpy(), [1, 2, 0])

        dose_to_save = sparse_vector_function(dose_pred_gy)
        dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                               columns=['data'])
        dose_df.to_csv('{}/{}.csv'.format('checkpoints/results/', pat_id))

        #dose_score = (torch.sum(torch.abs((dose * 100) - (pred * 100))) / torch.sum(mask))

        #log_dict = {'val_loss': dose_score, 'train_dose_score': dose_score}

        return #{'val_loss': dose_score, 'log': log_dict, 'progress_bar': log_dict}


    def test_epoch_end(self, outputs):
        submission_dir = make_directory_and_return_path('checkpoints/submissions')
        shutil.make_archive('{}/{}'.format(submission_dir, 'submission'), 'zip',
                            'checkpoints/results')
        return #{'val_loss_mean': val_loss_mean, 'log': log_dict, 'progress_bar': log_dict}


def main(args):

    model = DosePrediction(args)

    name = f'Unet_{args.used_act}_use_resnet_{str(args.use_SE)}_norm_{args.norm}_{str(args.num_layers)}layer_loss_use_mask_{str(args.use_mask)}'
    wandblogger = WandbLogger(name=name, project='doseprediction',
                              entity='lfetty')


    #checkpoint_callback = ModelCheckpoint(
    #    save_top_k=10,
    #    monitor='loss_mean',
    #    mode='min',
    #    prefix='',
    #    save_weights_only=True
    #)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.ngpus, logger=wandblogger)#, resume_from_checkpoint=f'checkpoints/{name}.pt')
    trainer.fit(model)
    trainer.save_checkpoint(f'checkpoints/{name}.pt')
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--training_data_dir', type=str, default='data/train-pats')
    parser.add_argument('--validation_data_dir', type=str, default='data/validation-pats-no-dose')
    parser.add_argument('--test_data_dir', type=str, default='data/test-pats')
    parser.add_argument('--num_train_pats', type=int, default=200)
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers used of the 3D feature loss. 0=no feature loss.')
    parser.add_argument('--loss_weights', nargs='+',type=int, default=[100, 50, 50, 50], help='weighting of the different feature layers')
    parser.add_argument('--use_mask', type=bool, default=False, help='use mask for feature loss')
    parser.add_argument('--down_samples', type=int, default=7,
                        help='how often the image dimensions get reduced. 7 downsamples for a 128x128x128 matrix is in the bottleneck 1x1x1.')
    parser.add_argument('--norm', type=str, default='instance', help='instance|batch|group')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='true weight decay of AdamW')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--use_resnet', type=bool, default=False, help='use resnet blocks between down/up convolutions.')
    parser.add_argument('--used_act', type=str, default='ReLU', help='Mish|ReLU')
    parser.add_argument('--ngf', type=int, default=64, help='filter number of first layer')

    args = parser.parse_args()

    main(args)
