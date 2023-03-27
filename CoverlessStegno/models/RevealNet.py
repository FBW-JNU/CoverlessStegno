import torch.nn as nn


class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward(self, input,secret_input=None):
        output=self.main(input)
        reveal_loss = self.get_reveal_loss(secret_input,output)
        return reveal_loss,output

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, mean, std)

    def get_reveal_loss(self,input,output):
        criterion = nn.MSELoss().cuda()
        mse_reveal = criterion(output,input)
        return mse_reveal

