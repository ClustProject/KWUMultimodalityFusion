# Semi-supervised Graph Convolutional Network based Emotion Recognition using Multimodal Biosignals
#### Produced by NeuroAI Lab. @Kwangwoon Univ.

생체 신호 기반 감정인식 기술은 정밀한 감정 관련 특징을 내재한 생체 신호의 특성 덕분에 높은 성능을 보여왔으나 복잡한
측정 장비로 인해 실용성이 적다고 평가받아왔다. 본 연구에서는, 감정 인식 성능 향상과 더불어 높은 실용성도 보장하기 위해
멀티미디어 시청자들로부터 취득된 공용 다중모달 DEAP 데이터 세트에서 3종의 생체신호(뇌전도, 피부전도도, 광용적맥파)를
활용한 준지도 그래프 신경망 기반 감정 인식 모델을 제안한다. 각 신호에서 실제 상용화된 웨어러블 기기로 측정 가능한 채널만
을 활용하고, 적은 양의 레이블링 된 데이터를 활용하는 준지도 학습 기법을 도입하여 실용성을 보장한다. 3종의 경량화된 생체신
호에 내재된 서로 다른 감정 관련 특징의 상호보완적인 특성을 융합하여 고수준의 특징을 합성할 수 있는 다중모달리티 융합
기법과 준지도 학습에 특화된 그래프 컨볼루션 신경망을 활용함으로써 실용성 뿐만 아니라 높은 감정 인식 성능 (97.37%)을 달성
하였다. 제안한 모델은 최신식의 (state-of-the-art) 준지도 학습 기반 뇌전도 감정 인식 기술 뿐만 아니라 레이블링된 데이터를
많이 사용하는 최신의 지도 학습 기반 생체 신호 감정 인식 모델들의 성능을 능가한다. 


# 1. MultimodalityFusion
A method to fuse similarity networks computed from different types of time-series data

## Similarity Network Fusion
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/d3710b0c-e0ba-4d54-a2d3-aabf516d3381){: width="10%" height="10%"}
*	각 신호로부터 DE, PSD 특징을 추출한 후, 모든 특징 벡터들을 Concatenate융합하여 1개의 특징 행렬을 생성한다.
*	각 특징 벡터간 유사성 점수를 계산하여, 6개의 유사성 행렬을 생성한다.
*	각 신호의 DE, PSD 유사성 행렬을 우선적으로 융합한다. (국소 유사성 융합)
*	이후, 국소 융합 행렬들을 전체적으로 융합한다.
*	각 유사성 행렬들은 하나의 그래프 구조를 나타내며, 유사성 행렬 융합은 6개의 그래프 구조를 하나의 그래프 구조로 융합한 것을 의미한다.



## Used signals (On a public DEAP dataset)


### 뇌전도, Electroencephalogram (EEG)
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/2a3ac3d9-cee3-426d-9af3-1492e9b0ea59)

*	뇌전도를 기록하기 위해서 전극을 두피에 부착하며, 전극의 수에 따라 뇌전도 신호의 채널 수가 정해진다.
*   DEAP dataset의 총 뇌전도 채널 수는 32개이다.
*	뇌전도 각 채널로부터 특정 뇌 지역의 활동을 알 수 있으며, 해당 뇌 지역이 활성화되는 경우 스파이크와 함께 복잡한 파형이 기록된다.
*	심전도, 심탄도 등의 심장 신호와 달리 반복되는 파형이 기록되지 않고 다른 생체 신호보다 복잡한 파형을 가지기 때문에 정밀하고 다양한 분석 기법이 필요하다.
*	뇌전도는 총 5개의 주파수 대역 (δ (1-3 Hz) wave, θ (4-7 Hz) wave, α (8-13 Hz) wave, β (14-30 Hz) wave, γ (31-50 Hz) wave)으로 나누어 질 수 있다.
*   DEAP dataset에서는 감마파를 제외한 총 4개의 주파수 대역만을 활용한다.


### 광용적맥파, Photoplethysmogram (PPG)
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/cf93b7d2-01b4-4b3b-bc09-a3549f1f4e89)
*	심장박동에 동기되어 손가락 말단의 혈관에서의 혈량이 증가하고 줄어드는 상태가 반복된다.
*   DEAP dataset의 총 광용적맥파 채널 수는 1개이다.
*	광용적맥파를 기록하기 위해 손가락 말단에 빛을 조사한 후 빛의 굴절, 흡수, 반사 등을 활용한다.
*	조사한 빛이 광수신기에 도달하여, 심장박동에 의한 혈류, 혈량 변화 등을 측정한다.
*	심장 박동 시 혈류량 변화가 급격하여 광용적맥파 신호에 Peak로 기록되며, 통상적으로 AC 성분을 광용적맥파라고 부른다.
*	감정적 흥분 시, 교감신경이 흥분하게 되면, 혈관은 수축하고 심장 박동은 빨라지기 때문에 놀라움, 분노, 흥분 등의 감정적 특징이 피부전도도에 기록될 수 있다.



### 피부전도도, Galvanic Skin Response (GSR)
![image](https://github.com/ClustProject/KWUMultimodalityFusion/assets/74770095/3a3b6783-8582-4b6d-a075-eaf4376a0a6f)
*	피부에 미세 전류를 흘려보내 두 지점 사이 (두 손가락 마디 등)의 전기 전도도를 측정한다.
*   DEAP dataset의 총 피부전도도 채널 수는 1개이다.
*	피부의 땀샘이 열리면, 전도도가 높아지며, 신호에서는 높은 Peak가 기록된다.
*	피부의 땀샘은 교감신경계의 통제하에 있기 때문에 피부의 전기전도도는 심리적, 신체적 각성상태를 측정하는 데에 활용될 수 있다.
*	특히, 감정적 각성과 교감신경 활동 사이의 관계에 대한 연구가 활발히 진행되고 있으며, 공포, 분노, 놀라움 등의 감정은 민감한 피부전도도 반응을 유발하여, 피부전도도는 거짓말 탐지기에 많이 활용된다.


### Used Frequency-domain features
* 시계열 데이터의 각 주파수 대역의 스펙트럼 복잡성을 정량화하는 Differential Entropy (DE) 및 각 주파수 대역의 스펙트럼 세기를 측정하는 Power Spectral Density (PSD)를 활용한다.


### Data seperation
* 전체 데이터의 9.52% 만큼만 train set으로 활용하고, 나머지는 test set으로 활용하는 준지도 학습 수행

### Summary
*   DEAP dataset의 총 3개의 생체 시계열 모달리티의 전체 채널 수는 34개이며, 그 중 뇌전도 채널이 32개로 뇌전도가 중심 모달리티이며, 광용적맥파 및 피부전도도는 보조 모달리티로서 활용한다.


# 2. Emotion Recognition

![image](https://github.com/KimDyun/Graph-Representation-Learning/assets/74770095/b9988835-7108-4a63-90be-6d33f6aa361c)

* DEAP dataset에서는 Valence-Arousal 감정 모델링 기반으로 감정을 구분한다.
* Valence는 감정의 긍정, 부정을 나타내며, Arousal은 감정의 세기 (흥분, 차분)을 나타낸다.
* 본 연구에서는 감정 모델링의 중심 점을 기준으로 High arousal (흥분), Low arousal (차분) 2진 분류 및 High valence (긍정), Low valence (부정) 2진 분류를 수행한다.



# 3.  MSGCN
Multimodal Semi-supervised Graph Convolutional Network (MSGCN)


*	모델 구조는 두개의 GCN으로 구성되며, GCN은 각 노드의 특징을 직접 연결된 주변 노드들로부터 엣지 가중치 (유사성 점수)와 곱해진 특징을 전파 받아 고수준의 특징으로 합성한다. 유사한 노드끼리 특징 합성이 이루어지므로 클래스 별 특징 군집을 뚜렷하게 구성할 수 있다는 장점이 있다.
*	Classifier로는 하나의 Fully-connected (FC) layer를 활용한다.

![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/fddc4631-3b5f-4ffa-bb65-9630a1a1c787)
*Graph Convolution*

