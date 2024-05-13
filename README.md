# Diagnosis of behavioral abnormalities from CCTV video

This project is defined in order to detect abnormal behaviors such as (running, falling, climbing the wall, etc.) in surveillance cameras and for different buildings and centers; This project has the ability to increase all types of anomalies.
This project has the ability to detect anomalies in offline video and in real time.
![Alt text](https://github.com/MoeinHan/Diagnosis-of-behavioral-abnormalities/blob/main/1.png)
![Alt text](https://github.com/MoeinHan/Diagnosis-of-behavioral-abnormalities/blob/main/2.png)

### Features
- Code can run on CPU
- Video/WebCam/External Camera/IP Stream Supported
- Recognize 5 action include walk, run, sit, fall and climb

## Requirements

* python 3.8

## How to Install

```bash
git clone https://github.com/MoeinHan/Diagnosis-of-behavioral-abnormalities.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Use

```bash
python PS_2Xopenvino.py 
```

introduce options:

* \--poseweights : model path(s)
* \--source : video path/0 for webcam/rtsp protocol/ make dataset you should pass file path
* \--device : cpu/0,1,2,3(gpu)
* \--view-img : display results. use 'store_true' if you want to store.
* \--store-video : store video in export file. use 'store_true' if you want to store.
* \--TransformerModel : Transformer model for inference time. It is the model for action recognition.
* \--line-thickness : bounding box thickness (pixels)
* \--Sport-mode : use in sporty env . use 'store_true' if you are in sporty mode.
* \--Factory-mode : use in factory env. use 'store_true' if you are in factory mode.

### Steps to run Code

- Run the code with mentioned command below.
```
#For CPU
python PS_2Xopenvino.py --poseweights "./models/yolov8m-pose_openvino_int8_model/yolov8m-pose.xml" --source "your custom video.mp4" --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'

#For View-Image
python PS_2Xopenvino.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source "your custom video.mp4" --view-img --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python PS_2Xopenvino.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source "your IP Camera Stream URL" --view-img --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'

#For WebCam
python PS_2Xopenvino.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source 0 --view-img --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'

#For External Camera
python PS_2Xopenvino.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source 1 --view-img --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'

#For store export offline video
python PS_2Xopenvino.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source "your custom video.mp4"  --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml' --store-video

# For multicamera action recognition
python PS_2XopenvinoMultiCamera.py --poseweights "./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml" --source "your custom video.mp4"  --TransformerModel './models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml' --store-video

```
## Support

Merge requests are welcome as a support for this project.