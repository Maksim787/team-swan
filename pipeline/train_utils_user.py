import torch


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(model, optimizer, criterion, loader):
    device = get_device()
    for epoch in range(1):
        train_epoch(model, optimizer, criterion, loader, device)


def train_epoch(model, optimizer, criterion, loader, device=torch.device('cpu')):
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
