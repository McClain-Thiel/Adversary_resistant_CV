from SE import SE_ResNet
import torch.optim
import torch.nn as nn
from Data import TinyImNet
from torch.utils.data import DataLoader


NUM_EPOCHS = 100

params = { 'batch_size': 64,
           'shuffle': True,
           'num_workers': 6
            }

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

    for epoch in range(NUM_EPOCHS):
        curr_loss, val_loss = 0.0, 0.0
        for (loc_batch, loc_label), (val_batch, val_label) in zip(train_gen, val_gen):
            inputs, labels = loc_batch.to(device), loc_label.to(device)
            opt.zero_grad()

            outputs = model.forward(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            val_batch, val_label = val_batch.to(device), val_label.to(device)

            val_output = model.forward(val_batch)
            val_loss += crit(val_output, val_label)
            curr_loss += loss.item()
        print("Loss for Epoch ", epoch, ': ', curr_loss)
        print("Val Loss for Epoch ", epoch, ': ', val_loss)

    torch.save(model.state_dict(), "model/resnet")

if __name__ == '__main__':
    main()


