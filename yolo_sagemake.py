import cv2
import boto3
import json
import time

# 1) SageMaker Runtime 클라이언트
sm_runtime = boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
endpoint_name = "yolov8-pytorch-2025-03-11-10-37-49-694013" 

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
