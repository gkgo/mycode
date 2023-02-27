import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import tqdm
import time
import os
import sys
import json
from resnet import resnet12,ConvNet4


GPU = torch.cuda.is_available()


def train(epoch, model, device, train_loader, optimizer):
    model.train()
    size = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
    
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f"tarin: {epoch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, device, test_loader):
    model.eval()
    correct = 0
    best_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        if correct > best_acc:
            best_acc = correct
    print('accuracy=',100. * correct/len(test_loader.dataset))



def main():
    device = torch.device("cuda" if GPU else "cpu")
    batch_size = 32
    nw = 2
#     model = resnet12().to(device)
    model = ConvNet4().to(device)
    # model = resnet12yuan().to(device)
    # model = resnet18().to(device)
    # model = resnet18gai().to(device)
    optimizer = optim.SGD(model.parameters(), lr =0.01,momentum=0.9,nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,70], gamma=0.05)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root,"datasets", "dataset", "flower_data")  # flower data set path
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])



    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices1.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    test_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    # train_data = torchvision.datasets.CIFAR100('./dataset',
    #                     transform=transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),  # 在水平方向上随机翻转
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]), download=False)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # test_data = torchvision.datasets.CIFAR100('./dataset',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  train=False, download=False)
    # test_loader = torch.utils.data.DataLoader(test_data)


    for epoch in range(80):
        start_time = time.time()
        train(epoch, model, device, train_loader, optimizer)
        test(model, device, test_loader)
        lr_scheduler.step()
        epoch_time = time.time() - start_time
        print(f'[ log ] roughly {(80 - epoch) / 3600. * epoch_time:.2f} h left\n')
    


if __name__ == '__main__':
    main()
