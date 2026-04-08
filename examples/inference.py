"""
室内障碍物检测 - 推理示例
Indoor Obstacle Detection - Inference Example
"""

from ultralytics import YOLO
import cv2
import argparse


def detect_image(model_path: str, image_path: str, conf: float = 0.5):
    """检测单张图片"""
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, conf=conf)
    
    print(f"\n检测结果 ({image_path}):")
    print("-" * 40)
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            print(f"  {class_name}: {confidence:.1%} @ [{x1}, {y1}, {x2}, {y2}]")
    
    return results


def detect_camera(model_path: str, conf: float = 0.5):
    """实时摄像头检测"""
    model = YOLO(model_path)
    
    print("启动摄像头检测... 按 'q' 退出")
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(source=frame, conf=conf, verbose=False)
        
        # 绘制检测结果
        annotated_frame = results[0].plot()
        
        cv2.imshow('Obstacle Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def detect_video(model_path: str, video_path: str, conf: float = 0.5):
    """检测视频文件"""
    model = YOLO(model_path)
    results = model.predict(source=video_path, save=True, conf=conf)
    print(f"视频检测完成！结果保存在 runs/detect/ 目录")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='室内障碍物检测')
    parser.add_argument('--model', type=str, default='weights/best.pt', help='模型路径')
    parser.add_argument('--source', type=str, default='0', help='图片/视频路径，或 0 表示摄像头')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    if args.source == '0':
        detect_camera(args.model, args.conf)
    elif args.source.endswith(('.mp4', '.avi', '.mov')):
        detect_video(args.model, args.source, args.conf)
    else:
        detect_image(args.model, args.source, args.conf)
