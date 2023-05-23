class CNN(nn.Module):

    # define init method
    # define layers
    def __init__(self, hiddenLayer=100):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, hiddenLayer)
        self.out = nn.Linear(hiddenLayer, 10)
        self.act = nn.ReLU()

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 6 * 14 * 14)
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x


class CNN2(nn.Module):

    def __init__(self, hiddenLayer=100):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5)
        self.fc1 = nn.Linear(24 * 10 * 10, hiddenLayer)
        self.out = nn.Linear(hiddenLayer, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = x.view(-1, 24 * 10 * 10)
        x = self.act(self.fc1(x))
        x = self.out(x)

        return x