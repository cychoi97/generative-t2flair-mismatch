#@title Autoload all modules
# %load_ext autoreload
# %autoreload 2

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets


#@title Visualization code

def image_grid(x):
  height = config.data.image_size
  width = config.data.image_size * 3
  channels = 1
  img = x.reshape(-1, height, width, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, height, width, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * height, w * width, channels))
  return img


def save_samples(x, idx, path):
  for b in range(x.size(0)):
    x_1 = x.permute(0, 2, 3, 1).detach().cpu().numpy()[b,:,:,0]
    x_2 = x.permute(0, 2, 3, 1).detach().cpu().numpy()[b,:,:,1]
    x_3 = x.permute(0, 2, 3, 1).detach().cpu().numpy()[b,:,:,2]
      
    y = np.concatenate([x_1, x_2, x_3], 1)
    img = y
    # img = image_grid(y)

    save_path = os.path.join(path, f'{idx:03}_{b:02}.png')

    plt.imsave(save_path, img, cmap='gray')


# @title Load the score-based model
num_checkpoint = 14
result_folder = "result"
from configs.ve import t1t2flair as configs
ckpt_filename = f"./{result_folder}/checkpoints/checkpoint_{num_checkpoint}.pth"
config = configs.get_config()  
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

batch_size = 64 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
sample_num = 1 # total sampling num = batch size * sample num
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
save_path = './result/generated_images'
os.makedirs(save_path, exist_ok=True)

for idx in range(0, sample_num):
  sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                        inverse_scaler, snr, n_steps=n_steps,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        eps=sampling_eps, device=config.device)

  x, n = sampling_fn(score_model)
  save_samples(x, idx, save_path)