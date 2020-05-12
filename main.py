from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import sys

import os
'''
a = np.loadtxt('plot.txt')
fig, ax1 = plt.subplots()
l1 = ax1.plot(a[:, 0], a[:, 1], label='Training Loss',color = 'r')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss')
ax2 = ax1.twinx()
l2 = ax2.plot(a[:, 0], np.ones(30)-a[:, 2], label='Validation Error')
l = l1 + l2
labs = [ln.get_label() for ln in l]
ax2.set_ylabel('Validation Error')
ax2.legend(l, labs, loc='best')
plt.show()

exit()
'''
'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(9, 9, 3, 1)
        self.conv3 = nn.Conv2d(9, 9, 3, 1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.2)
        self.batchnorm1 = nn.BatchNorm2d(9)
        self.batchnorm2 = nn.BatchNorm2d(9)
        self.batchnorm3 = nn.BatchNorm2d(9)
        self.fc1 = nn.Linear(81, 64)
        self.fc2 = nn.Linear(64, 10)


    def feature_vector(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.relu(x)
        return output

    def forward(self, x):
        x = self.feature_vector(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))



def closest_neighbor(feature_vec, i0):
    distance = np.sum((feature_vec - feature_vec[i0,:])**2,axis=1)
    order = distance.argsort()
    neighbors = order[:9]
    for i in range(9):
        print(neighbors[i], distance[neighbors[i]])
    return neighbors


'''
feature_vec = np.loadtxt('feature_vector.txt')
neighbors = closest_neighbor(feature_vec,99)
test_dataset = datasets.MNIST('../data', train=False,transform=transforms.ToTensor())
all_pic = np.zeros((28,28*9))
for i in range(9):
    all_pic[:,i*28:(i*28+28)] = np.array(test_dataset[neighbors[i]][0][0])

plt.imshow(all_pic)
plt.show()
exit()
'''




def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    feature_vec = torch.Tensor([])
    colors = ['b','g','r','c','m','y','k','w','orange','lavender']
    labels = []
    #confusion_matrix = np.zeros((10,10)).astype(int)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            feature = model.feature_vector(data)           # Get feature vector
            feature_vec = torch.cat((feature_vec,feature))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            ### Get confusion matrix
            true_labels = np.array(target.view_as(pred)).astype(int)
            #pred = np.array(pred).astype(int)
            for i in range(len(true_labels)):
                labels.append(colors[true_labels[i][0]])

                #confusion_matrix[true_labels[i],pred[i]]+=1
                #if pred[i] != true_labels[i]:
                    #plt.imshow(np.array(data[i][0]))
                    #plt.show()
                    #print(pred[i])
            test_num += len(data)

    #with open('labels.txt','w') as f:
        #f.write(' '.join(_ for _ in labels))
        #f.close()

    #exit()
    #embedding = TSNE(n_components=2)
    #feature_vec = embedding.fit_transform(feature_vec)
    #np.savetxt('embedded_feature_vector.txt',np.array(feature_vec))

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))



def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--augmentation', type = str, default = 'none', metavar = 'N', help='choose type of data augmentation: none, rotation, translation')

    parser.add_argument('--training-number', type = float, default = 1.0, metavar = 'N', help='Decide subset of training set to be trained')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))
        ###
        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))



        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return


    ### Data Augmentation
    if args.augmentation != 'none':
        print('Data Augmentation Performed:', args.augmentation, '\n')
        if args.augmentation == 'rotation':
            transformation =transforms.Compose([transforms.RandomRotation(45, fill=(0,)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        elif args.augmentation == 'hflip':
            transformation = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transformation)
    val_dataset = datasets.MNIST('../data', train=True,
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.



    ### Extract 15 % from each class in the training set as validation set
    subset_indices_train = []
    subset_indices_valid = []
    labels = [[] for _ in range(10)]
    training_subset = int(len(train_dataset) * args.training_number)
    for i in range(training_subset):
        item = train_dataset[i]
        labels[item[1]].append(i)
    for i in range(10):
        order = np.array(labels[i])
        np.random.shuffle(order)
        split = int(0.85*len(order))
        for j in range(split):
            subset_indices_train.append(order[j])
        for j in range(split,len(order),1):
            subset_indices_valid.append(order[j])


    print(train_dataset)
    exit()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop

    plotter = np.zeros((args.epochs,3))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        plotter[epoch-1, 1:3] = test(model, device, val_loader)
        plotter[epoch-1,0] = epoch
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
