# Fashion_MNIST

회사에서 위암 T병기 예측을 위해 CNN을 돌릴 필요가 있었는데 그 전에 CNN에 대해 실습해보고자 했던 Fashion MNIST 코드다.

Fashion MNIST 는 Classification Task고, 위암 T병기 예측은 Segmentation Task 기 때문에 완전히 같다고 할 수 는 없지만, 연습용 코드를 한줄한줄 짜고 이해한 덕에 수월한 프로젝트 완수를 할 수 있었다. 

# Result

![Accuracy](https://user-images.githubusercontent.com/29745280/147676168-2c635a23-5e0c-4114-a2c6-ced8e5a32f69.png)

![Loss](https://user-images.githubusercontent.com/29745280/147676162-e158fa1c-1d2e-4694-9371-4d2e851c8446.png)

| Class | Accuracy |  precision | recall | f1-score |
|---|---|---|---|---|
| T-shirt/Top | 86.50% | 0.85 | 0.85 | 0.85 | 
| Trouser | 99.70%  |  0.98     |   0.99    |    0.98  | 
| Pullover |  86.70%   |   0.86   |     0.85    |    0.86 | 
| Dress |  89.50%   |   0.89    |    0.93    |    0.91 | 
| Coat |  93.60%  |   0.86    |    0.85    |    0.86 | 
| Sandal |  97.50%  |  0.98    |    0.97    |    0.97  | 
| Shirt |  68.90%  |   0.75    |    0.73    |    0.74   | 
| Sneaker |  92.30%   |   0.94    |    0.95    |    0.94  | 
| Bag |  97.60%   |   0.98    |    0.98    |    0.98  | 
| Ankle Boot  |  98.80%   |  0.95     |   0.96     |   0.96 |

## Accuracy : 91%

# 딥러닝 학습 향상을 위한 고려 사항
+ 가중치 감소(Weight Decay)
  + 오버피팅을 최소화하는 방법
+ 데이터 증강(Data Augmentation)
  + 오버피팅 해소는 물론 훈련 및 테스트 데이터에 대한 정확도를 높일 수 있는 방법
+ 가중치 초기화(Weight Initialization)
  + 기울기 소실(Gradient Vanishing) 방지
+ 학습률 스케쥴러(Learning Rate Scheduler)
  + 학습이 진행되면서 학습률을 그 상황에 맞게 가변적으로 적당하게 변경을 통해 더 낮은 손실값 얻음
+ 학습 데이터의 정규화(Data Normalization)
  + 입력되는 데이터에 대해서 공간상 분포를 정규화를 통해 더 높은 정확도
+ 다양한 경사하강법(Gradient Descent Variants)
  + SDG, Adam ..
+ 배치 정규화(Batch Normalization)
  + 각 신경망의 활성화 값 분포가 적당히 퍼지도록 개선하여 원할한 학습이 진행되도록 돕는 기법
+ 드롭아웃(Drop Out)
  + 정확도가 떨어지지만 오버피팅을 억제하기 위한 기법 
