import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

###############################################################################

class AverageMeter(object):
    # computes and stores the average and current value
    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

###############################################################################

def save_model(file_path, file_name, model, optimizer=None):
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))
    
###############################################################################

def load_model(ckpt_path, model, optimizer=None):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

###############################################################################

def accuracy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
    
###############################################################################

def train(train_loader, val_loader, model, model_name, epochs, learning_rate, gamma, step_size, device, load_saved_model, ckpt_save_freq, ckpt_save_path, ckpt_path, model_path, report_path):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
         model, optimizer = load_model(ckpt_path=ckpt_path, model=model, optimizer=optimizer)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"
        
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True)
        for batch_idx, (images, labels) in loop_train:
            images = images.to(device)
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1 = accuracy(labels_pred, labels)
            top1_acc_train.update(acc1[0], images.size(0))
            loss_avg_train.update(loss.item(), images.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_top1_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - epoch {epoch}/{epochs}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True)
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer)

        model.eval()
        mode = "val"

        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True)
            for batch_idx, (images, labels) in loop_val:
                optimizer.zero_grad()
                images = images.to(device).float()
                labels = labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_val.update(acc1[0], images.size(0))
                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_top1_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_top1_acc_till_current_batch":top1_acc_val.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"Valid - epoch {epoch}/{epochs}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True)
        lr_scheduler.step()

    torch.save(model.state_dict(), f"{model_path}/{model_name}.pt")
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    
    return model, optimizer, report

###############################################################################

def PlotReport(report):
    
    train_report = report[report['mode']=='train']
    val_report = report[report['mode']=='val']
    train_report_for_plot = train_report[train_report.batch_index == train_report['batch_index'].max()]
    val_report_for_plot = val_report[val_report.batch_index == val_report['batch_index'].max()]

    fig, axs = plt.subplots(1,2,figsize=(8,2.5))

    axs[0].set_title("Train and Validation Loss\n", y=0.95, fontsize=8)
    axs[0].plot(train_report_for_plot["avg_train_loss_till_current_batch"].values, linewidth=2, label="Train")
    axs[0].plot(val_report_for_plot["avg_val_loss_till_current_batch"].values, linewidth=2, label="Validation")
    axs[0].set_xlabel("Epoch", fontsize=7) ; axs[0].set_ylabel("Loss", fontsize=7)
    axs[0].grid(axis="y", alpha=0.5) ; axs[0].legend(loc=0, prop={"size": 7})
    axs[0].tick_params(axis="x", labelsize=7) ; axs[0].tick_params(axis="y", labelsize=7)

    axs[1].set_title("Train and Validation Accuracy\n", y=0.95, fontsize=8)
    axs[1].plot(train_report_for_plot["avg_train_top1_acc_till_current_batch"].values, linewidth=2, label="Train")
    axs[1].plot(val_report_for_plot["avg_val_top1_acc_till_current_batch"].values, linewidth=2, label="Validation")
    axs[1].set_xlabel("Epoch", fontsize=7) ; axs[1].set_ylabel("Loss", fontsize=7)
    axs[1].grid(axis="y", alpha=0.5) ; axs[1].legend(loc=0, prop={"size": 7})
    axs[1].tick_params(axis="x", labelsize=7) ; axs[1].tick_params(axis="y", labelsize=7)

    plt.show()

###############################################################################

def PlotPred(dataset, model, dict_path, number=3, mean=[0.49139968,0.48215827,0.44653124], std=[0.24703233,0.24348505,0.26158768]):

    model.load_state_dict(torch.load(dict_path))
    model.eval()

    pred_loader = torch.utils.data.DataLoader(dataset, batch_size=number, shuffle=True)
    inputs, classes = next(iter(pred_loader))
    
    outputs=model(inputs)
    _, preds = torch.max(outputs, 1)
    preds=preds.cpu().numpy()
    classes=classes.numpy()

    for i in range(number):
      img = inputs[i].detach().numpy().transpose((1, 2, 0))
      img = np.array(std)*img + np.array(mean)
      img = img.clip(0,1)
      fig, ax = plt.subplots(1, figsize=(1,1))
      ax.imshow(img)
      ax.set_axis_off() ; plt.show()
      print(f'Class: {classes[i]}\nPredicted: {preds[i]}')