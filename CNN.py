import torch.nn as nn
class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    '''in_channels(int): input image의 channel수 . rgb 이미지라면 3 .

    out_channels(int): convolution에 의해서 생성된 channel의 수

    kernel_size(int or tuple): convoling_kenel 의 크기. 보통은 filter라고 부르는 것과 동일

    stride(int or tuple): convolution의 stride를 얼만큼 줄 것이가 .Default는 1 

    padding(int or tuple): zero padding을 input의 양쪽에 인자 만큼 해준다.Default는 0 이라서 기본적으로 설정해주지 않으면 zero padding은 하지 않음.'''

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out.view(-1, 64*6*6)
        out = out.view(out.size(0), -1)#16,(64*6*6)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out