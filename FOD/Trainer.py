from os import replace
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
from tqdm import tqdm

from FOD.utils import get_loss, get_optimizer
from FOD.FocusOnDepth import FocusOnDepth

import DPT.util.io
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        self.model = FocusOnDepth(
                    (3,config['Dataset']['transforms']['resize'],config['Dataset']['transforms']['resize']),
                    patch_size=config['General']['patch_size'],
                    emb_dim=config['General']['emb_dim'],
                    resample_dim=config['General']['resample_dim'],
                    read=config['General']['read'],
                    nhead=config['General']['nhead']
        )

        # self.model = DPTDepthModel(
        #     path=config['Dataset']['paths']['model_dpt'],
        #     backbone="vitl16_384",
        #     non_negative=True,
        #     enable_attention_hooks=False,
        # )

        #self.model.half()
        self.model.to(self.device)
        # print(self.model)
        # exit(0)

        self.loss = get_loss(config)
        self.optimizer = get_optimizer(config, self.model)

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']
        if self.config['wandb']['enable']:
            wandb.init(project="FocusOnDepth", entity=self.config['wandb']['username'])
            wandb.config = {
                "learning_rate": self.config['General']['lr'],
                "epochs": epochs,
                "batch_size": self.config['General']['batch_size']
            }
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            self.model.train()
            for i, (x, y_depth) in tqdm(enumerate(train_dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                #x, y_depth = x.to(self.device).half(), y_depth.to(self.device).half()
                x, y_depth = x.to(self.device), y_depth.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimizer
                outputs_depth = self.model(x)

                outputs_depth = outputs_depth.squeeze(1)
                y_depth = y_depth.squeeze(1)

                # get loss
                loss = self.loss(outputs_depth, y_depth)
                loss.backward()

                # step optimizer
                self.optimizer.step()
                running_loss += loss.item()

                if self.config['wandb']['enable']:
                    wandb.log({"loss": loss.item()})

                if i%50 == 0:
                    print('epoch {} : loss = '.format(epoch+1), running_loss/(50*self.config['General']['batch_size']))
                    running_loss = 0
            self.run_eval(train_dataloader, val_dataloader)
        print('Finished Training')

    def run_eval(self, train_dataloader, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- train_dataloader -: torch dataloader
            :- val_dataloader -: torch dataloader
        """
        self.model.eval()
        val_loss = 0.
        validation_samples = ()
        truths_samples = ()
        preds_samples = ()

        with torch.no_grad():
            for i, (x, y_depth) in tqdm(enumerate(val_dataloader)):
                x, y_depth = x.to(self.device), y_depth.to(self.device)
                outputs_depth = self.model(x)

                outputs_depth = outputs_depth.squeeze(1)
                y_depth = y_depth.squeeze(1)

                # get loss
                loss = self.loss(outputs_depth, y_depth)
                val_loss += loss.item()

                if len(validation_samples) < self.config['wandb']['images_to_show']:
                    validation_samples = (*validation_samples, x[0].unsqueeze(0))
                    truths_samples = (*truths_samples, y_depth[0].unsqueeze(0).unsqueeze(0))
                    preds_samples = (*preds_samples, outputs_depth[0].unsqueeze(0).unsqueeze(0))

            val_loss = val_loss / len(val_dataloader)
            print('val_loss = ', val_loss)

            if self.config['wandb']['enable']:

                wandb.log({"val_loss": val_loss})

                imgs = torch.cat(validation_samples, dim=0).detach().cpu().numpy()
                imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

                truths_tensor = torch.cat(truths_samples, dim=0).detach().cpu().numpy()
                truths = np.repeat(truths_tensor, 3, axis=1)

                val_tensor = torch.cat(preds_samples, dim=0).detach().cpu().numpy()
                preds = np.repeat(val_tensor, 3, axis=1)
                preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)

                print("******************************************************")
                print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
                print(truths.shape, truths.mean().item(), truths.max().item(), truths.min().item())
                print(preds.shape, preds.mean().item(), preds.max().item(), preds.min().item())
                print("******************************************************")

                imgs = imgs.transpose(0,2,3,1)
                truths = truths.transpose(0,2,3,1)
                preds = preds.transpose(0,2,3,1)

                #val_predictions = np.concatenate((truth, pred), axis=-2).transpose(0,2,3,1)
                #output_dim = (2*int(self.config['wandb']['im_h']), int(self.config['wandb']['im_w']))
                output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

                wandb.log(
                    {"img": [wandb.Image(cv2.resize(im, output_dim), caption='val_{}'.format(i+1)) for i, im in enumerate(imgs)],
                    "imgTruths": [wandb.Image(cv2.resize(im, output_dim), caption='val_truths{}'.format(i+1)) for i, im in enumerate(truths)],
                    "imgPreds": [wandb.Image(cv2.resize(im, output_dim), caption='val_pred{}'.format(i+1)) for i, im in enumerate(preds)]}
                )
