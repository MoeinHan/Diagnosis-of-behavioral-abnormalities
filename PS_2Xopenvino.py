import os
import torchvision.transforms
from ultralytics.utils import ops
import csv
import operator
from collections import deque
import cv2
import time
import torch
import argparse
import numpy as np
from torchvision import transforms
import ipywidgets as widgets
from openvino.runtime import Core


from utils.datasets import letterbox
from utils.plots import plot_fps_time_comparision, colors,plot_one_box_kpt, plot_status_box

def model_utils(model,image,frame_width):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    # image = letterbox(image, auto=False)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = torchvision.transforms.Resize((480,640))(image)
    # image = image.to('cuda')  # convert image data to device
    # image = image.float()  # convert image to float precision (cpu)

    output_data = model(image)
    output_data = output_data[model.output(0)]
    nms_kwargs = {"agnostic": False, "max_det": 80}
    preds = ops.non_max_suppression(
        torch.from_numpy(output_data),
        0.6,
        0.4,
        nc=1,
        **nms_kwargs
    )
    return output_data, preds, image


@torch.no_grad()
def run(poseweights="./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml",source="football1.mp4",device='cpu',view_img=False, store_video=False, Sport_mode=False, Factory_mode=False,
        line_thickness=3,TransformerModel='./models/MiT_500_98_6_A5_V3_justpose/MiT_500_98_6_A5_V3_justpose.xml'):

    
    model_path = './vitpose-s-coco_25.pth'
    yolo_path = './yolov5s.pt'

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps

    anomaly_counter = 0
    core = Core()
    if os.path.exists('conf_list.csv'):
        os.remove('conf_list.csv')
    header = ['climb', 'fall', 'walk', 'run', 'sit']
    with open('conf_list.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
    device2 = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )

    core = Core()
    pose_ov_model = core.read_model(poseweights)
    if device2.value != "CPU":
        pose_ov_model.reshape({0: [1, 3, 480, 640]})
    # device_name = "GPU"
    pose_compiled_model = core.compile_model(model=pose_ov_model, device_name=device2.value)

    core = Core()
    TransformerModel = core.read_model(TransformerModel)
    if device2.value != "CPU":
        TransformerModel.reshape({0: [1, 25,34]})
    TransformerModel = core.compile_model(TransformerModel, device2.value)

    if source or source.isnumeric():

        if source.isnumeric(): # check if use webcam
            source = int(source)

        # read video from many source like (Webcam, offline, surveilliance camera(rtsp))
        cap = cv2.VideoCapture(source)  # pass video to videocapture object
        if (cap.isOpened() == False):  # check if videocapture not opened
            print('Error while trying to read video. Please check path again')
            raise SystemExit()

        else:
            frame_width = 480  # get video frame width
            frame_height = int(cap.get(4))  # get video frame height
            vid_write_image = letterbox(cap.read()[1], auto=False)[0]  # init videowriter
            resize_height, resize_width = vid_write_image.shape[:2]
            if store_video:
                # build video writer object for store video export
                files = sorted(os.listdir('./exports'))
                if len(files) != 0 :
                    files = files[-1]
                    name = int(files.split('_')[-1][:-4]) + 1
                else:
                    name = 0
                fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                out = cv2.VideoWriter(f"./exports/exp_{name:004}.avi",
                                      fourcc, cap.get(cv2.CAP_PROP_FPS),
                                    (int(cap.get(3)), int(cap.get(4))))

                # gst_out = ('appsrc caps=video/x-raw,format=GRAY8,width=1280,height=800,framerate=30/1 ! '
                #            'videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! '
                #            'h265parse ! filesink location=test.h265 ')

            counter = -1
            # conf_list_for_plot_realtime_status_figure = deque([], maxlen=90000)
            main_list = deque([], maxlen=20000) # build empty Qeue for Store detected objects
            label_qeue = deque([], maxlen=5) # store labels to detect anomaly and showing in Status box

            with open('conf_list.csv', 'a') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                # writer.writeheader()
                while (cap.isOpened):  # loop until cap opened or video not complete
                    status_plot_mean_list = deque([], maxlen=200)
                    tm = cv2.TickMeter()
                    print("Frame {} Processing".format(frame_count + 1))
                    counter += 1
                    ret, frame = cap.read()  # get frame and success from video capture
                    label_list_4_anomaly_detection = deque([], maxlen=1000)
                    Now = "Unknown" #defult status
                    anomaly_from_texture = 0
                    if ret:  # if success is true, means frame exist
                        orig_image = frame  # store frame
                        start_time = time.time()  # start time for fps calculation
                        tm.start()
                        output_data, preds, image = model_utils(pose_compiled_model,orig_image,frame_width)

                        _, _, H, W = image.shape
                        im0 = transforms.Resize((resize_height,resize_width))(image[0])
                        im0 = im0.permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                        im0 = im0.cpu().numpy().astype(np.uint8)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
                        im0 = cv2.resize(im0, (resize_width,resize_height), cv2.INTER_LINEAR)

                        n = 0
                        for i, result in enumerate(output_data):  # detections per image

                            if len(preds):  # check if no pose
                                n = len(preds[0])
                                print("No of Objects in Current Frame : {}".format(n))
                                for det_index, (*xyxy, conf, cls) in enumerate(preds[0][:, :6]):  # loop over poses for drawing on frame and give objects keypoints to Transformer model

                                    kpts = preds[0][det_index,6:]
                                    if (len(kpts) == 51) and conf >=0.7:
                                        row = kpts
                                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                                        row = [float(j) for k, j in enumerate(row) if k % 3 == 0 or k % 3 == 1]
                                        row = [float("{:.3f}".format(z / W)) if k % 2 == 0 else float("{:.3f}".format(z / H)) for k, z in enumerate(row)]
                                        occured_label_index = 10 # get random number for initialize variable
                                        row.append(x1)
                                        row.append(y1)
                                        row.append(x2)
                                        row.append(y2)

                                        if counter == 0: # check for first iteration
                                            main_list.append([[row,occured_label_index]])

                                        else:
                                            flag = 0
                                            for h, r in enumerate(main_list): # Loop over object that stored in main_list and match new keypoint row with Eclidean distance
                                                # we use Eclidean distance for similarity search
                                                num1 = np.array(r[-1][0][:-4])
                                                num2 = np.array(row[:-4])
                                                sub = np.subtract(num1,num2)
                                                dist = np.linalg.norm(list(sub))
                                                if dist <= 0.9 and flag == 0:
                                                    r.append([row, occured_label_index])
                                                    flag = 1
                                                    if len(r) == 25: # check if number of seq for Transformer model is ok

                                                        skeleton = np.expand_dims(np.array(list(map(lambda its: its[:][0][:-4], r))), axis=0).astype(np.float32)

                                                        prediction = TransformerModel(skeleton) # prediction Transformer model
                                                        # prediction = TransformerModel.run(output_names, {TransformerModel.get_inputs()[0].name: skeleton})
                                                        index, num = max(enumerate(prediction[0][0]),
                                                                         key=operator.itemgetter(1))
                                                        status_vector = np.zeros(6)
                                                        status_vector[index] += num

                                                        status_plot_mean_list.append(status_vector)

                                                        # assign string label according prediction index and score
                                                        if index == 0 and num >= 0.85:
                                                            r[-1][1] = index
                                                            label = 'climb'

                                                        elif index == 1 and num >= 0.30:
                                                            label = 'walk'
                                                            r[-1][1] = index

                                                        elif index == 2 and num >= 0.80:
                                                            label = 'fall'
                                                            r[-1][1] = index

                                                        elif index == 3 and num >= 0.30:
                                                            label = 'run'
                                                            r[-1][1] = index
                                                            Now = "Normal"

                                                        elif index == 4 and num >= 0.30:
                                                            label = 'sit'
                                                            r[-1][1] = index
                                                        # elif index == 5 and num >= 0.30:
                                                        #     label = 'stand'
                                                        #     r[-1][1] = index
                                                        else:
                                                            label = f'person {h}'
                                                        # clean each object stack for prevent memory overflow
                                                        # if len(r) >= 10000:
                                                        #     r = r[-2000:]
                                                        #
                                                        # if index == 0 and num >= 0.3:
                                                        #     r[-1][1] = index
                                                        #     label = 'walk'
                                                        #
                                                        # elif index == 1 and num >= 0.70:
                                                        #     label = 'climb'
                                                        #     r[-1][1] = index
                                                        #
                                                        # elif index == 2 and num >= 0.80:
                                                        #     label = 'fall'
                                                        #     r[-1][1] = index
                                                        #
                                                        # elif index == 3 and num >= 0.60:
                                                        #     label = 'run'
                                                        #     r[-1][1] = index
                                                        #     Now = "Normal"
                                                        #
                                                        # elif index == 4 and num >= 0.60:
                                                        #     label = 'sit'
                                                        #     r[-1][1] = index
                                                        #
                                                        # elif index == 5 and num >= 0.4:
                                                        #     label = 'down'
                                                        #     r[-1][1] = index
                                                        # else:
                                                        #     label = f'person {h}'
                                                        # # # clean each object stack for prevent memory overflow
                                                    #########################################################
                                                        # if len(r) >= 10000:
                                                        #     r = r[-2000:]
                                                        #
                                                        # if index == 0 and num >= 0.3:
                                                        #     r[-1][1] = index
                                                        #     label = 'walk'
                                                        #
                                                        # elif index == 1 and num >= 0.70:
                                                        #     label = 'climb'
                                                        #     r[-1][1] = index
                                                        #
                                                        # elif index == 2 and num >= 0.80:
                                                        #     label = 'fall'
                                                        #     r[-1][1] = index
                                                        #
                                                        # elif index == 3 and num >= 0.60:
                                                        #     label = 'run'
                                                        #     r[-1][1] = index
                                                        #     Now = "Normal"
                                                        #
                                                        # elif index == 4 and num >= 0.60:
                                                        #     label = 'sit'
                                                        #     r[-1][1] = index
                                                        # #
                                                        # # elif index == 5 and num >= 0.6:
                                                        # #     label = 'stand'
                                                        # #     r[-1][1] = index
                                                        # else:
                                                        #     label = f'person {h}'
                                                        # # clean each object stack for prevent memory overflow
                                                    ##############################################################
                                                        # garbage collector
                                                        # if len(r) >= 10000:
                                                        #     r = r[-2000:]
                                                        label_list = np.array(r, dtype='object')[:,1] # collecting occured action of each object

                                                        if Sport_mode: # Ignore climb for anomaly detection
                                                            if (2 in label_list):
                                                                if 15 in label_list:
                                                                    r[-1][1] = 15
                                                                    Now = "Anomaly Occure"
                                                                else:
                                                                    anomaly_counter += 1
                                                                    label_qeue.append(label)
                                                                    r[-1][1] = 15
                                                                    Now = "Anomaly Occure"
                                                            else:
                                                                Now = "Normal"

                                                        elif Factory_mode: # track 4 actions except walking
                                                            if (0 in label_list) or (2 in label_list) or (3 in label_list) or (4 in label_list):
                                                                if 15 in label_list:
                                                                    r[-1][1] = 15
                                                                    Now = "Anomaly Occure"
                                                                else:
                                                                    anomaly_counter += 1
                                                                    r[-1][1] = 15
                                                                    label_qeue.append(label)
                                                                    Now = "Anomaly Occure"
                                                            else:
                                                                Now = "Normal"

                                                        else: # Normal anomaly detection include Climbing and Falling
                                                            if (0 in label_list) or (2 in label_list):
                                                                if 15 in label_list:
                                                                    r[-1][1] = 15
                                                                    Now = "Anomaly Occure"
                                                                else:
                                                                    anomaly_counter += 1
                                                                   
                                                                    r[-1][1] = 15
                                                                    label_qeue.append(label)
                                                                    Now = "Anomaly Occure"
                                                            else:
                                                                Now = "Normal"

                                                        label_list_4_anomaly_detection.append(label)
                                                        if view_img:
                                                            plot_one_box_kpt(np.array(r[-1][0][-4:]), im0, label=label, color=colors(0, True),
                                                                             line_thickness=line_thickness, kpt_label=False,
                                                                             kpts=kpts,
                                                                             steps=3,
                                                                             orig_shape=im0.shape[:2])
                                                        if store_video:
                                                            box = np.array(r[-1][0][-4:])

                                                            x_0 = int((box[0]/im0.shape[1])*orig_image.shape[1])
                                                            y_0 = int((box[1]/im0.shape[0])*orig_image.shape[0])
                                                            x_1 = int((box[2]/im0.shape[1])*orig_image.shape[1])
                                                            y_1 = int((box[3]/im0.shape[0])*orig_image.shape[0])

                                                            box = [x_0, y_0, x_1, y_1]
                                                            plot_one_box_kpt(np.array(box), orig_image, label=label, color=colors(0, True),
                                                                             line_thickness=line_thickness, kpt_label=False,
                                                                             kpts=kpts,
                                                                             steps=3,
                                                                             orig_shape=orig_image.shape[:2])
                                                        main_list[h] = r[1:] # make empty space for next prediction
                                            if flag == 0:
                                                main_list.append([[row, occured_label_index]])

                            # search for anomaly detection in specific area
                            if not(Sport_mode) or not(Factory_mode):
                                if len(label_list_4_anomaly_detection) != 0:
                                    run_count = label_list_4_anomaly_detection.count("run")
                                    walk_count = label_list_4_anomaly_detection.count("walk")
                                    len_list = len(label_list_4_anomaly_detection)
                                    walk_percent = (walk_count/len_list)
                                    run_percent = (run_count/len_list)
                                    if walk_percent > 0.55:
                                        if run_percent >= 0.1:
                                            anomaly_from_texture = 1
                                            Now = "Anomaly Occure"
                                            label_qeue.append("run")
                                        else:
                                            Now = "Normal"
                            if view_img:
                                # Draw Status Box
                                plot_status_box(im0, anomaly_detect=anomaly_counter, no_person=n, Factory_mode=Factory_mode,
                                                            Sport_mode=Sport_mode, anomaly_from_texture=anomaly_from_texture, Now=Now, log_anomaly=label_qeue)
                            if store_video:
                                # Draw Status Box on Original image size
                                plot_status_box(orig_image, anomaly_detect=anomaly_counter, no_person=n, Factory_mode=Factory_mode,
                                                            Sport_mode=Sport_mode, anomaly_from_texture=anomaly_from_texture, Now=Now, log_anomaly=label_qeue)

                            if len(status_plot_mean_list) == 0:
                                status_plot_mean_list.append([0, 0, 0, 0, 0])

                            row = np.mean(status_plot_mean_list, axis=0)
                            row = {'climb':row[0], 'fall':row[2], 'walk':row[1], 'run':row[3], 'sit':row[4]}
                            writer.writerow(row)
                            tm.stop()
                            print(f"Average FPS: {tm.getFPS():.3f}")

                            end_time = time.time()  # Calculatio for FPS
                            fps = 1 / (end_time - start_time)
                            total_fps += fps
                            frame_count += 1

                            fps_list.append(total_fps)  # append FPS in list
                            time_list.append(end_time - start_time)  # append time in list

                            # Stream results
                            if view_img:
                                cv2.imshow("Anomaly Detection Demo", im0)
                                # cv2.moveWindow('Anomaly Detection Demo', 0, 0)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            if store_video:
                                out.write(orig_image)  #writing the video frame


                    else:
                        break

            avg_fps = total_fps / frame_count
            print(f"Average FPS: {avg_fps:.3f}")

            # plot the comparision graph
            # plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)
            cap.release()
            out.release()
            cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', default='./models/yolov8s-pose_openvino_int8_model/yolov8s-pose.xml', help='model path(s)')
    parser.add_argument('--source', type=str, default='/home/oem/DadePardaz/pose_estimation/yolov8-pose-estimation/videotest/h2.mp4', help='video path/0 for webcam/rtsp protocol/ make dataset you should pass file path') #video source
    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')   #device arugments for yolo algo
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--store-video', action='store_true', help='Store video export') # store video in export file
    parser.add_argument('--TransformerModel', default='./models/CONV_180_98_17_V4_4actionOK/CONV_180_98_17_V4_4action.xml', help='Transformer model for inference time') # model for action recognition
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--Sport-mode',action='store_true' , help='use in sporty env') # when Environment is sporty
    parser.add_argument('--Factory-mode',action='store_true' , help='use in factory env') # when Environment is factory

    opt = parser.parse_args()
    return opt

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    # strip_optimizer(opt.device,opt.poseweights)
    main(opt)
