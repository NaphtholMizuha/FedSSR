import torch.nn as nn

class CnnGray(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[120, 84], out_dim=10, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6 , 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], out_dim),         
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))

# class Cnn(nn.Module):
#     def __init__(self, input_dim=400, hidden_dims=[120, 84], out_dim=10, dropout=0.5):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 6 , 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dims[0]),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dims[1], out_dim),         
#         )
        
#     def forward(self, x):
#         return self.fc(self.conv(x))

class Cnn(nn.Module):
    def __init__(self, input_dim=400, hidden_dims=[120, 84], out_dim=10, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
