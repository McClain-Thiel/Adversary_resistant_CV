from SE import SE_ResNet
import torch.optim
import torch.nn as nn
from Data import TinyImNet
from torch.utils.data import DataLoader


NUM_EPOCHS = 3

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

    for epoch in range(NUM_EPOCHS):
        curr_loss = 0.0
        for loc_batch, loc_label in train_gen:
            inputs, labels = loc_batch.to(device), loc_label.to(device)
            opt.zero_grad()

            outputs = model.forward(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            curr_loss += loss.item()
        print("Loss for Epoch ", epoch, ': ', curr_loss)

if __name__ == '__main__':
    main()


