
import torch
import pandas as pd

#%% Dataloader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# from dataloaders.dataloaders import DataLoader_Manual
# DIR_TRAIN="./datasets/CIFAR10/train/"
# DIR_VAL="./datasets/CIFAR10/val/"
# (train_dataset, val_dataset) = DataLoader_Manual(DIR_TRAIN, DIR_VAL)

from dataloaders.dataloaders import DataLoader_Torch
ROOT="./datasets"
(train_dataset, val_dataset) = DataLoader_Torch(ROOT)

#%% Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import yaml
with open('utils/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['learning_rate']
gamma = config['gamma']
step_size = config['step_size']
ckpt_save_freq = config['ckpt_save_freq']

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%% Basic Model Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from nets.nets import BasicModel, InceptionModel, ResnextModel
from deeplearning.deeplearning import train, PlotReport, PlotPred

trainer = train(
    train_loader=train_loader,
    val_loader=val_loader,
    model = BasicModel(),
    model_name="Test",
    epochs=epochs,
    learning_rate=learning_rate,
    gamma = gamma,
    step_size = step_size,
    device=device,
    load_saved_model=False,
    ckpt_save_freq=ckpt_save_freq,
    ckpt_save_path="./saved/checkpoints",
    ckpt_path="./",
    model_path="./saved/models",
    report_path="./saved/reports")

report = pd.read_csv("./saved/reports/Test_report.csv")
PlotReport(report)
PlotPred(dataset=val_dataset, model=BasicModel(), dict_path='./saved/models/Test.pt')
