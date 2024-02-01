# 1. KWU-Analysis-Model
Deep learning model for physiological time-series analysis

## Content
### Analysis-Model (Discriminative Graph Transformer Model)
#### Produced by NeuroAI Lab. @Kwangwoon Univ.

![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/baff03da-0fca-4b2e-a8ea-ab6382a8ff6d)
*Discriminative Graph Transformer Model (DTGM)*

*	EEG 감정인식 모델은 Graph Encoder Module (GEM), 그리고 Graph Transformer Module (GTM)으로 구성된다.
*	GEM은 두개의 GCN으로 구성되며, GCN은 각 노드의 특징을 직접 연결된 주변 노드들로부터 엣지 가중치 (유사성 점수)와 곱해진 특징을 전파 받아 고수준의 특징으로 합성한다. 유사한 노드끼리 특징 합성이 이루어지므로 클래스 별 특징 군집을 뚜렷하게 구성할 수 있다는 장점이 있지만, 직접 연결된 노드의 특징만 고려하므로, 국소적인 이웃 관계에 고착된다.
* 이 문제를 보완하고자, 그래프 트랜스포머를 도입하였으며, 기존 어텐션 스코어 (cosine 유사성) 계산 시 엣지 정보 (유클리드 거리 기반 유사성)를 추가로 활용함으로써 그래프 전체적 이웃 관계를 고려한다.
*	Classifier로는 하나의 Fully-connected (FC) layer를 활용한다.

![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/fddc4631-3b5f-4ffa-bb65-9630a1a1c787)
*Graph Convolution*
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/e72caeb8-e972-41ee-81f8-94153514f012)
*Graph Multi-head Attention*


## Used signals

### Electroencephalogram (EEG)
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/2a3ac3d9-cee3-426d-9af3-1492e9b0ea59)

*	뇌전도를 기록하기 위해서 전극을 두피에 부착하며, 전극의 수에 따라 뇌전도 신호의 채널 수가 정해진다.
*	뇌전도 각 채널로부터 특정 뇌 지역의 활동을 알 수 있으며, 해당 뇌 지역이 활성화되는 경우 스파이크와 함께 복잡한 파형이 기록된다.
*	심전도, 심탄도 등의 심장 신호와 달리 반복되는 파형이 기록되지 않고 다른 생체 신호보다 복잡한 파형을 가지기 때문에 정밀하고 다양한 분석 기법이 필요하다.
*	뇌전도는 총 5개의 주파수 대역 (δ (1-3 Hz) wave, θ (4-7 Hz) wave, α (8-13 Hz) wave, β (14-30 Hz) wave, γ (31-50 Hz) wave)으로 나누어 질 수 있다.


# 2. KWUMultimodalityFusion
A method to fuse similarity networks computed from different types of time-series data

## Similarity Network Fusion
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/d3710b0c-e0ba-4d54-a2d3-aabf516d3381)
*	각 신호로부터 DE, PSD 특징을 추출한 후, 모든 특징 벡터들을 Concatenate융합하여 1개의 특징 행렬을 생성한다.
*	각 특징 벡터간 유사성 점수를 계산하여, 6개의 유사성 행렬을 생성한다.
*	각 신호의 DE, PSD 유사성 행렬을 우선적으로 융합한다. (국소 유사성 융합)
*	이후, 국소 융합 행렬들을 전체적으로 융합한다.
*	각 유사성 행렬들은 하나의 그래프 구조를 나타내며, 유사성 행렬 융합은 6개의 그래프 구조를 하나의 그래프 구조로 융합한 것을 의미한다.



## Used signals

### Electroencephalogram (EEG)
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/2a3ac3d9-cee3-426d-9af3-1492e9b0ea59)

*	뇌전도를 기록하기 위해서 전극을 두피에 부착하며, 전극의 수에 따라 뇌전도 신호의 채널 수가 정해진다.
*	뇌전도 각 채널로부터 특정 뇌 지역의 활동을 알 수 있으며, 해당 뇌 지역이 활성화되는 경우 스파이크와 함께 복잡한 파형이 기록된다.
*	심전도, 심탄도 등의 심장 신호와 달리 반복되는 파형이 기록되지 않고 다른 생체 신호보다 복잡한 파형을 가지기 때문에 정밀하고 다양한 분석 기법이 필요하다.
*	뇌전도는 총 5개의 주파수 대역 (δ (1-3 Hz) wave, θ (4-7 Hz) wave, α (8-13 Hz) wave, β (14-30 Hz) wave, γ (31-50 Hz) wave)으로 나누어 질 수 있다.


### Photoplethysmogram (PPG)
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/cf93b7d2-01b4-4b3b-bc09-a3549f1f4e89)
*	심장박동에 동기되어 손가락 말단의 혈관에서의 혈량이 증가하고 줄어드는 상태가 반복된다.
*	광용적맥파를 기록하기 위해 손가락 말단에 빛을 조사한 후 빛의 굴절, 흡수, 반사 등을 활용한다.
*	조사한 빛이 광수신기에 도달하여, 심장박동에 의한 혈류, 혈량 변화 등을 측정한다.
*	심장 박동 시 혈류량 변화가 급격하여 광용적맥파 신호에 Peak로 기록되며, 통상적으로 AC 성분을 광용적맥파라고 부른다.
*	감정적 흥분 시, 교감신경이 흥분하게 되면, 혈관은 수축하고 심장 박동은 빨라지기 때문에 놀라움, 분노, 흥분 등의 감정적 특징이 피부전도도에 기록될 수 있다.



### Galvanic Skin Response (GSR)
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/3a3b6783-8582-4b6d-a075-eaf4376a0a6f)
*	피부에 미세 전류를 흘려보내 두 지점 사이 (두 손가락 마디 등)의 전기 전도도를 측정한다.
*	피부의 땀샘이 열리면, 전도도가 높아지며, 신호에서는 높은 Peak가 기록된다.
*	피부의 땀샘은 교감신경계의 통제하에 있기 때문에 피부의 전기전도도는 심리적, 신체적 각성상태를 측정하는 데에 활용될 수 있다.
*	특히, 감정적 각성과 교감신경 활동 사이의 관계에 대한 연구가 활발히 진행되고 있으며, 공포, 분노, 놀라움 등의 감정은 민감한 피부전도도 반응을 유발하여, 피부전도도는 거짓말 탐지기에 많이 활용된다.



