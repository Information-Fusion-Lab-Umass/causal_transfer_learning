import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.w = args.w
        self.h = args.h
        self.input_channels = args.input_channels
        self.p = 0

        self.convs = nn.Sequential(nn.Conv2d(self.input_channels, 8, kernel_size=3, stride=2, padding = self.p), nn.ReLU(),
                                   nn.Conv2d(8, 16, kernel_size=3, stride= 1, padding = self.p), nn.ReLU(),
                                   nn.Conv2d(16, 16, kernel_size=3, stride= 1, padding = self.p), nn.ReLU())
        self.convw = self.conv_size_out(self.conv_size_out(self.conv_size_out(self.w, 3, 2), 3, 1), 3, 1)
        self.convh = self.conv_size_out(self.conv_size_out(self.conv_size_out(self.w, 3, 2), 3, 1), 3, 1)
        self.conv_output_size = int(self.convw) * int(self.convh) * 16
        self.fc_h_z_a1 = nn.Linear(self.conv_output_size, self.hidden_size)
        self.fc_h_z_a2 = nn.Linear(self.conv_output_size, self.hidden_size)
        self.fc_h_z_a3 = nn.Linear(self.conv_output_size, self.hidden_size)
        self.fc_h_z_a4 = nn.Linear(self.conv_output_size, self.hidden_size)

        self.fc_h_a1 = nn.Linear(self.hidden_size, 1)
        self.fc_h_a2 = nn.Linear(self.hidden_size, 1)
        self.fc_h_a3 = nn.Linear(self.hidden_size, 1)
        self.fc_h_a4 = nn.Linear(self.hidden_size, 1)


    def conv_size_out(self, w, k, s):
        return ((w - k+(2*self.p))/s) + 1

    def forward(self, x, trial = 1, log=False):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        z1 = F.relu(self.fc_h_z_a1(x)) # hidden layer
        z2 = F.relu(self.fc_h_z_a2(x))
        z3 = F.relu(self.fc_h_z_a3(x))
        z4 = F.relu(self.fc_h_z_a4(x))

        a1 = F.relu(self.fc_h_a1(z1))
        a2 = F.relu(self.fc_h_a2(z2))
        a3 = F.relu(self.fc_h_a3(z3))
        a4 = F.relu(self.fc_h_a4(z4))

        result = torch.cat((a1,a2,a3,a4), dim = 1)
        result = result.view(-1, self.action_space)
        return result, (z1,z2,z3,z4)
