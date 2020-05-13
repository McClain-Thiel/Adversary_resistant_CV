from SE import SE_ResNet
import torch.optim
import torch.nn as nn
from Data import TinyImNet
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


NUM_EPOCHS = 100

params = { 'batch_size': 64,
           'shuffle': True,
           'num_workers': 6
            }

def calc_acc(outputs, labels):
    max_vals, max_indices = torch.max(outputs, 1)
    n = max_indices.size(0)  # index 0 for extracting the # of elements
    train_acc = (max_indices == labels).sum(dtype=torch.float32) / n
    return train_acc

def main():
    model = SE_ResNet(num_classes=200).get_model()
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    crit = nn.CrossEntropyLoss()

    train_set = TinyImNet('tiny-imagenet-200/train', True, 'tiny-imagenet-200/train',)
    train_gen = DataLoader(train_set, **params)

    val_set = TinyImNet('tiny-imagenet-200/val', False, 'tiny-imagenet-200/train',)
    val_gen = DataLoader(val_set, **params)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    print("Don't compare val loss to train loss, they are on different scales.")
    train_acc_arr, val_acc_arr = [], []
    for epoch in range(NUM_EPOCHS):
        curr_loss, val_loss = 0.0, 0.0
        for (loc_batch, loc_label), (val_batch, val_label) in zip(train_gen, val_gen):
            inputs, labels = loc_batch.to(device), loc_label.to(device)
            opt.zero_grad()

            print("Starting forward pass for epoch: ", epoch)
            outputs = model.forward(inputs)
            print("Starting backward pass for epoch: ", epoch)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            val_batch, val_label = val_batch.to(device), val_label.to(device)

            val_output = model.forward(val_batch)
            val_loss += crit(val_output, val_label)
            curr_loss += loss.item()

        print("Loss for Epoch ", epoch, ': ', curr_loss)
        print("Val Loss for Epoch ", epoch, ': ', val_loss)

        (train_inputs, train_labesl),  (val_inputs, val_labels) = next(train_gen), next(val_gen)
        train_outputs, val_outputs = model.forward(train_inputs), model.forward(val_inputs)
        train_acc, val_acc = calc_acc(train_outputs, train_labesl), calc_acc(val_outputs, val_labels)
        print("Epoch: ", epoch, " Train Accuracy: ", train_acc, " Test Accuracy: ", val_acc)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)


    data = {"epoch":np.arange(0,100),
            "train_acc": train_acc_arr,
            "val_acc": val_acc_arr}

    df = pd.DataFrame(data=data)
    df.to_csv('data.csv')

    torch.save(model.state_dict(), "model/resnet")

if __name__ == '__main__':
    main()


