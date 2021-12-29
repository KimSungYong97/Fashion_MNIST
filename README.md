# Fashion_MNIST

회사에서 위암 T병기 예측을 위해 CNN을 돌릴 필요가 있었는데 그 전에 CNN에 대해 실습해보고자 했던 Fashion MNIST 코드다.

Fashion MNIST 는 Classification Task고, 위암 T병기 예측은 Segmentation Task 기 때문에 오나전히 같다고 할 수 는 없지만, 연습용 코드를 한줄한줄 짜고 이해한 덕에 수월한 프로젝트 완수를 할 수 있었다. 

# Result

![Accuracy](https://user-images.githubusercontent.com/29745280/147642209-8e11540e-6b57-4c89-b8f9-798940b3681b.png)

![Loss](https://user-images.githubusercontent.com/29745280/147642213-c4d7ca31-1eeb-4317-85f1-dfade041214b.png)

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

## Accuracy : 0.91%

# 비고

성능이 막 뛰어나다고 할 수 는 없다.
