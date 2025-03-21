- 일반적으로 사용하려면 프리 티어로 로그인 하면 SageMaker 2개월 무료 사용가능함!
- 실습 링크 : https://aws.amazon.com/ko/getting-started/hands-on/machine-learning-tutorial-generate-predictions-without-writing-code/

![image.png](attachment:a96c75f9-7721-4946-a8f0-7a0583d6e19a:image.png)

![image.png](attachment:9befb5d7-dc07-45a2-b931-dfc23e911f13:image.png)

![image.png](attachment:9b3ca3dd-7ade-4eca-b851-4b3b98d394ed:image.png)

### My Models

1. My Models 탭으로 이동한 후 모델 생성
2. 데이터 선택
    1. shipping-logs.csv 데이터 생성 → 배송 도착 날짜 예측 모델 생성

![image.png](attachment:65125979-cd8a-46d3-87d8-cb872363944c:image.png)

### 데이터에서 예측 열 선택(예측할 Target)

![image.png](attachment:85f09bdd-9e69-4b33-8972-9d9724b22a5f:image.png)

![image.png](attachment:6f80f0b6-06a3-4cd5-8cc0-1bdf4fa2d437:image.png)

### 예측 모델 생성

예측 모델은 Numeric model type으로 선택 → Target Column이 Numeric이기 때문

![image.png](attachment:6b918dd1-eb85-4e7f-853b-5fefdd83ce2e:image.png)

![image.png](attachment:3f9c83e5-4476-4527-a574-2522fab0946e:image.png)

### Model Preview

![image.png](attachment:f57d72e9-7b70-40ba-ac9d-6ddeafd13e9f:image.png)

### Quick build

약 3분 소요

![image.png](attachment:003d5638-8171-446a-9a1e-f83211f60aeb:image.png)

![image.png](attachment:a1a891fe-60ae-4bf7-b9cf-51c3957d81b5:image.png)

![image.png](attachment:1bf9c2a9-9a3b-4a73-8cbc-1520947d46f4:image.png)

### Predict Single

![image.png](attachment:2fb2c143-b261-401a-8649-6b7d779acb53:image.png)

![image.png](attachment:29cb3767-c109-4d57-9eb4-0b46cd52e43e:image.png)

배송 도착 기간 : 약 14일로 예측

### Predict Batch

![image.png](attachment:4565a314-8e0e-4842-9771-7d90c4f3a263:image.png)

---

## SageMaker, CloudFormation을 활용한 Yolo 배포

SageMaker : 머신 러닝 모델을 구축, 학습 및 배포하는 프로세스를 간소화하는 Amazon Web Services(AWS)의 머신 러닝 서비스

![image.png](attachment:aaf5bb7f-8da0-42cc-a88f-633b9a6ca75d:image.png)

![image.png](attachment:62582b3e-c073-47df-8262-18cd4299ebed:image.png)

1. IAM 설정
    1. Amazon SageMaker, AWS CloudFormation 및 Amazon S3에 필요한 권한이 있는 IAM 역할이 필요
2. WSL을 사용한 AWS CLI & AWS CDK 설정

## 1. GitHub 리포지토리 복제

1. **원격 리포지토리 복제**
    
    ```bash
    git clone https://github.com/aws-samples/host-yolov8-on-sagemaker-endpoint.git
    ```
    
2. **디렉터리 이동**
    
    ```bash
    cd host-yolov8-on-sagemaker-endpoint/yolov8-pytorch-cdk
    ```
    

이 리포지토리 안에는 **AWS CDK 스크립트**, **모델 배포를 위한 코드**, **SageMaker 구성 파일** 등이 포함되어 있습니다.

---

## 2. CDK 환경 설정 (Python)

1. **Python 가상 환경 생성 & 활성화**
2. **프로젝트 종속성 설치**
    
    ```bash
    pip3 install -r requirements.t
    ```
    
3. **AWS CDK 라이브러리 업그레이드(선택)**
    
    ```bash
    pip install --upgrade aws-cdk-lib
    ```
    
4. **CDK 구성 확인**
    - `cdk.json`, `app.py`, `cdk/` 폴더 등을 살펴본 후 프로젝트 구조 이해

---

## 3. CloudFormation 스택 생성 & 모델 배포

1. **CDK Synthesis**
    
    ```bash
    cdk synth
    ```
    
    - CDK 코드에서 CloudFormation 템플릿을 생성(로컬 출력)
2. **CDK Bootstrap**
    
    ```bash
    cdk bootstrap
    
    ```
    
    - 필요한 S3, ECR, IAM 리소스를 자동 세팅(처음 한 번만)
3. **CDK Deploy**
    
    ```bash
    cdk deploy
    ```
    
    - SageMaker Notebook 인스턴스, S3 버킷, IAM Role 등 **AWS 리소스**가 생성됨
    - 배포가 완료되면, AWS 콘솔 → SageMaker → Notebook 인스턴스에서 확인 가능

---

## 4. SageMaker 노트북에서 YOLO11 모델 엔드포인트 배포

### (A) 노트북 인스턴스 접속

- **AWS 콘솔** → **Amazon SageMaker** → **Notebook 인스턴스** → **Open Jupyter**
- JupyterLab/Jupyter 환경에 접근

### (B) `1_DeployEndpoint.ipynb` 예시 코드

```python
## 1.1 Import Python Libraries
import os, sagemaker, subprocess, boto3
from datetime import datetime
from sagemaker import s3
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorchModel
from sagemaker.deserializers import JSONDeserializer

## 1.2 Install YOLO11 / YOLOv8
!pip3 install ultralytics
from ultralytics import YOLO

model_name = 'yolov8l.pt'  # 예: YOLOv8l (YOLO11은 동일 구조 사용)
YOLO(model_name)

# Tar model + code/
bashCommand = f"tar -cpzf model.tar.gz {model_name} code/"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
process.communicate()

## 1.3 Upload to S3
s3_client = boto3.client('s3')
response = s3_client.list_buckets()
for b in response['Buckets']:
    if 'yolov8' in b["Name"]:
        bucket = 's3://' + b["Name"]
        break

sess = sagemaker.Session(default_bucket=bucket.split('s3://')[-1])
prefix = "yolov8/demo-custom-endpoint"
model_data = s3.S3Uploader.upload("model.tar.gz", bucket + "/" + prefix)

print(f'Model Data: {model_data}')
role = get_execution_role()

## 1.4 Create PyTorchModel
model = PyTorchModel(
    entry_point='inference.py',
    model_data=model_data,
    framework_version='1.12',
    py_version='py38',
    role=role,
    env={'TS_MAX_RESPONSE_SIZE':'20000000', 'YOLOV8_MODEL': model_name},
    sagemaker_session=sess
)

## 1.5 Deploy on SageMaker Endpoint
INSTANCE_TYPE = 'ml.m5.4xlarge'  # GPU: 'ml.g4dn.xlarge' / small CPU: 'ml.t3.medium'
ENDPOINT_NAME = 'yolov8-pytorch-' + datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
print(f'Endpoint Name: {ENDPOINT_NAME}')

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    deserializer=JSONDeserializer(),
    endpoint_name=ENDPOINT_NAME
)
print("Deploy Complete!")

```

- 위 노트북을 실행하면, SageMaker 엔드포인트가 생성됨 (예: `yolov8-pytorch-2025-03-11-07-47-44-182168`)

### (C) 테스트 노트북 (`2_TestEndpoint.ipynb`)

```python
python
복사
import boto3, json

endpoint_name = "yolov8-pytorch-2025-03-11-07-47-44-182168"
runtime_sm_client = boto3.client("sagemaker-runtime")

with open("test_image.jpg", "rb") as f:
    payload = f.read()

response = runtime_sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/x-image",
    Body=payload
)
result = response["Body"].read().decode('utf-8')
pred = json.loads(result)
print("Prediction:", pred)

```

- 엔드포인트로 이미지를 전송 → YOLO11 추론 결과(JSON)로 확인
- 시각화(마스크·바운딩박스 등) 가능

---

## 5. 로컬 PC에서 Webcam 실시간 추론

아래 스크립트를 **로컬**에서 실행 (전제: `aws configure`로 자격증명 설정 완료, `boto3` 설치)

```python
import cv2
import boto3
import json
import time

# 1) SageMaker Runtime 클라이언트
sm_runtime = boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
endpoint_name = "yolov8-pytorch-2025-03-11-07-47-44-182168" 

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    success, encoded_img = cv2.imencode(".jpg", frame)
    if not success:
        print("Failed to encode image")
        break

    payload = encoded_img.tobytes()

    # 4) 엔드포인트로 추론 요청
    start_time = time.time()
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image", 
        Body=payload
    )
    result_bytes = response["Body"].read()
    end_time = time.time()

    # 5) 결과 파싱
    result_str = result_bytes.decode('utf-8')
    try:
        pred = json.loads(result_str)
    except:
        pred = {}
        print("Invalid JSON response:", result_str)

    # 6) 바운딩 박스나 세그먼트 마스크 등 결과를 frame에 그리기
    if "boxes" in pred:
        boxes = pred["boxes"]
        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{score:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 7) 추론 시간 표시
    fps_info = f"Inference time: {(end_time - start_time)*1000:.0f} ms"
    cv2.putText(frame, fps_info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # 8) 화면에 표시
    cv2.imshow("SageMaker YOLO Realtime", frame)

    # 9) ESC 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

```

- 각 프레임을 엔드포인트로 전송해 YOLO 결과(바운딩박스)를 로컬 화면에 표시
- 네트워크+추론 지연 때문에 FPS는 낮을 수 있음(주로 데모용)
