from torchvision.datasets import MNIST
import torchvision
import torch.utils.data as Data
import torch
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Linear
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch import optim

import cv2

mnist = MNIST(root="./", transform=torchvision.transforms.ToTensor(), train=True, download=False)

train_loader = Data.DataLoader(dataset=mnist, batch_size=1, shuffle=True)
# for step, (b_x, b_y) in enumerate(train_loader):
#     # print(b_y)
#     # print(b_x)
#     # cv2.imshow("conv", b_x.data.numpy()[0][0])
#     # cv2.waitKey(0)
#     print("input shape", b_x.data.numpy()[0][0].shape)
#     n = net(input=b_x)
#     # print(n)
#     print("output shape", n.data.numpy()[0][0].shape)
#
#     # cv2.imshow("conv", n.data.numpy()[0][0])
#     # cv2.waitKey(0)


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = Sequential(
            Conv2d(1, 32, 3, 1, 2),
            ReLU(),
            Conv2d(32, 32, 3, 1, 2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            # Conv2d(32, 64, 3, 1, 2),
            # ReLU(),
            # Conv2d(64, 64, 3, 1, 2),
            # ReLU(),
            # MaxPool2d(kernel_size=2, stride=2),
            #
            # Conv2d(64, 128, 3, 1, 2),
            # ReLU(),
            # Conv2d(128, 128, 3, 1, 2),
            # ReLU(),
            # Conv2d(128, 128, 3, 1, 2),
            # ReLU(),
            # MaxPool2d(kernel_size=2, stride=2),
            #
            # Conv2d(128, 256, 3, 1, 2),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 2),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 2),
            # ReLU(),
            # MaxPool2d(kernel_size=2, stride=2),
            #
            # Conv2d(256, 256, 3, 1, 2),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 2),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 2),
            # ReLU()
        )
        self.out = Sequential(
            Linear(32 * 16 * 16, 10),
            Softmax()
        )

    def forward(self, x):
        conv = self.conv(x)
        print(conv.shape)
        flatten = conv.view(conv.size(0), -1)
        # print(flatten)
        out = self.out(flatten)
        return out, x


cnn = CNN()
print(cnn)  # net architecture


optimizer = optim.Adam(cnn.parameters(), lr=0.01)   # optimize all cnn parameters
loss_func = CrossEntropyLoss()


for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
    output = cnn(b_x)[0]               # cnn output
    # print(output)
    # print(b_y)
    loss = loss_func(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    if step % 50 == 0:
        test_output, last_layer = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
        accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
        print('train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
