import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import asyncio

class CNN(nn.Module):
    """ CNN model for image classification. """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7 * 7 * 64, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def load_model(model_path, device):
    """ 파일에서 훈련된 모델을 불러옵니다. """
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, device):
    """ 훈련된 모델을 사용하여 이미지의 레이블을 예측합니다.
    이미지는 회색조로 변환되고 28x28 픽셀로 크기 조정됩니다.
    """
    # 이미지를 불러오고 전처리합니다
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor()])
    input_image = transform(image).unsqueeze(0).to(device)

    # 이미지의 레이블을 예측합니다
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

async def predict_result():
    # 테스트 이미지의 레이블을 예측합니다
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('./CNN_MNIST', device)
    predicted_label = predict_image(model, './test.png', device)
    return predicted_label

if __name__ == "__main__":
    # 예측을 실행하고 결과를 출력합니다
    print(asyncio.run(predict_result()))