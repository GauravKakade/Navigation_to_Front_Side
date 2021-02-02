import cv2
import numpy as np
import time
from ensemble_boxes import *
import math
from vidgear.gears import NetGear
import imagiz

# Load Yolo
server=imagiz.Server()
options = {"flag": 0, "copy": False, "track": False}

client = NetGear(
    address="192.168.1.213",
    port="3456",
    protocol="tcp",
    pattern=0,
    receive_mode=True,
    logging=True,
    **options
)

netdepth = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
net = cv2.dnn.readNet("custom-yolov4-tiny-detector-7000-rgb.weights", "custom-yolov4-tiny-detector-rgb.cfg")


classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = netdepth.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in netdepth.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
while True:

    depth_colormap = client.recv()  # vidgear
    frame2 = depth_colormap
    message = server.receive()
    frame = cv2.imdecode(message.image, 1)
    frame_id += 1
    height, width, channels = frame2.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    blob2 = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    netdepth.setInput(blob)
    outsdepth = netdepth.forward(output_layers)
    net.setInput(blob2)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids_depth = []
    confidences_depth = []
    boxes_depth = []
    class_ids_rgb = []
    confidences_rgb = []
    boxes_rgb = []

    class_ids_depth_f = []
    confidences_depth_f = []
    boxes_depth_f = []
    class_ids_rgb_f = []
    confidences_rgb_f = []
    boxes_rgb_f = []
    boxes_list = []
    scores_list = []
    labels_list = []
    depth_detection = False
    rgb_detection = False

    front_side_detected = False
    left_side_detected = False
    rear_side_detected = False
    right_side_detected = False

    angle_to_corner1 = 0
    angle_to_corner2 = 0
    distance_to_corner1 = 0
    distance_to_corner2 = 0
    xx = 0
    yy = 0
    ww = 0
    hh = 0

    #for Depth detection
    for out in outsdepth:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            Class_id_for_matching1 = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes_depth.append([x, y, w, h])
                boxes_depth_f.append([x/1000, y/1000, ((x + w)/1000), ((y + h)/1000)])
                confidences_depth.append(float(confidence))
                class_ids_depth.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes_depth, confidences_depth, 0.6, 0.3)

    for i in range(len(boxes_depth)):
        if i in indexes:
            x, y, w, h = boxes_depth[i]
            label = str(classes[class_ids_depth[i]])
            confidence = confidences_depth[i]
            color = colors[class_ids_depth[i]]
            xx = x
            yy = y
            ww = w
            hh = h
            cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), color, 2)
            cv2.putText(depth_colormap, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 3)
            cv2.circle(depth_colormap, (int(x + w/2), int(y + h/3)), radius=15, color=(255, 255, 0), thickness=-1) #for Z depth

            cv2.circle(depth_colormap, (int(xx + ww/6), int(yy + hh/3)), radius=15, color=(255, 255, 0), thickness=-1) #for Corner1

            cv2.circle(depth_colormap, (int(xx + ww - ww/6), int(yy + hh/3)), radius=15, color=(255, 255, 0), thickness=-1) #for Corner2


            ##print("theta1 =", angle_to_corner1, "and theta2 =", angle_to_corner2)
            depth_detection = True


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(depth_colormap, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Depth", depth_colormap)


    # for RGB Detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            Class_id_for_matching2 = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                confidences_rgb.append(float(confidence))
                class_ids_rgb.append(class_id)
                boxes_rgb.append([x, y, w, h])
                boxes_rgb_f.append([x / 1000, y / 1000, ((x + w) / 1000), ((y + h) / 1000)])

    indexes = cv2.dnn.NMSBoxes(boxes_rgb, confidences_rgb, 0.6, 0.3)

    for i in range(len(boxes_rgb)):
        if i in indexes:
            x, y, w, h = boxes_rgb[i]
            label = str(classes[class_ids_rgb[i]])
            confidence = confidences_rgb[i]
            color = colors[class_ids_rgb[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 3)
            rgb_detection = True

    # For display of RGB
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("RGB", frame)
    elapsed_time = time.time() - starting_time
    boxes_list = [boxes_depth_f, boxes_rgb_f]
    #print(boxes_list)
    scores_list = [confidences_depth, confidences_rgb]
    labels_list = [class_ids_depth, class_ids_rgb]
    ##print(labels_list)
    #For weighted boxes fusion
    weights = [1, 3]
    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0)
    boxes_wbm = []
    boxes_wbm = boxes


    #For display of WBF
    if rgb_detection and depth_detection and Class_id_for_matching1 == Class_id_for_matching2:
        for box2 in boxes:
            color2 = (255, 0, 0)
            #print(int(boxes_wbm[0][0]*1000))
            cv2.rectangle(frame2, ((int(boxes_wbm[0][0]*1000)), (int(boxes_wbm[0][1]*1000))), ((int(boxes_wbm[0][2]*1000)), (int(boxes_wbm[0][3]*1000))), color2, 2)
            cv2.imshow("Weighted Boxes Fusion", frame2)
            depth = depth_frame.get_distance(int(xx + ww / 2), int(yy + hh / 3))
            distance_to_corner1 = round(depth_frame.get_distance(int(xx + ww/6), int(yy + hh/3)), 2)
            distance_to_corner2 = round(depth_frame.get_distance(int(xx + ww - ww/6), int(yy + hh/3)), 2)
            angle_to_corner1 = int(((int(xx + ww/6) - 320) / (320)) * (43))
            angle_to_corner2 = int(((int(xx + ww - ww/6) - 320) / (320)) * (43))
            print("distance_to_corner1", distance_to_corner1, "distance_to_corner2", distance_to_corner2)
            #print("Z-depth from camera surface to ", label, "side is", depth, "metres")

            if Class_id_for_matching1 == Class_id_for_matching2 == 0:
                front_side_detected = True
                #print("Front Side Detected")

            if Class_id_for_matching1 == Class_id_for_matching2 == 1:
                left_side_detected = True
                print("Left Side Detected")

            if Class_id_for_matching1 == Class_id_for_matching2 == 2:
                rear_side_detected = True
                print("Rear Side Detected")

            if Class_id_for_matching1 == Class_id_for_matching2 == 3:
                right_side_detected = True
                print("Right Side Detected")


    #Code if Front side detected
    if front_side_detected == True:
        print("theta1 =", angle_to_corner1, "and theta2 =", angle_to_corner2)
    #Case 1: If TB is at left of the Front side
        if angle_to_corner1 >= 0 and angle_to_corner2 >= 0:
            print("Front Case 1")
            width_of_box = 0.6
            print(width_of_box)
            theta3 = angle_to_corner2 - angle_to_corner1
            theta33 = (math.sin(math.radians(theta3)) / width_of_box) * distance_to_corner2
            print("theta3 :", theta3, "theta33 :", theta33)
            theta4 = 180 - math.asin(theta33) * 180 / (math.pi)

            theta5 = theta4 - 76
            # print(theta5)
            pr = math.cos(math.radians(theta5))
            # print(pr)
            distance_to_destination = math.sqrt(
                (distance_to_corner1 * distance_to_corner1) + (1.23 * 1.23) - 2 * distance_to_corner1 * 1.23 * (pr))
            # print(distance_to_destination)
            theta7 = 180 - theta3 - theta4
            theta8 = 76 - theta7
            pr2 = (math.sin(math.radians(theta8)) / distance_to_destination) * 1.23
            print(pr2)
            print("angle_to_corner1 =", angle_to_corner1, "angle_to_corner2 =", angle_to_corner2, "theta3 =", theta3,
                  "theta4 =", theta4, "theta5 =", theta5, "theta7 =", theta7, "theta8 =", theta8)
            beta = math.asin(pr2) * 180 / (math.pi)
            print("Turning by", beta, "degrees and travelling", distance_to_destination, "metres")

        #Case 2: If TB is at front of the Front side
        if angle_to_corner1 < 0 and angle_to_corner2 >= 0:
            print("Front Case 2")

        # Case 3: If TB is at right of the Front side
        if angle_to_corner1 < 0 and angle_to_corner2 >= 0:
            print("Front Case 3")
        #use same logic as for Front Case 1



    # Code if Left side detected
    if left_side_detected == True:
        print("theta1 =", angle_to_corner1, "and theta2 =", angle_to_corner2)

    # Code if rear side detected
    if rear_side_detected == True:
        print("theta1 =", angle_to_corner1, "and theta2 =", angle_to_corner2)

    # Code if right side detected
    if right_side_detected == True:
        print("theta1 =", angle_to_corner1, "and theta2 =", angle_to_corner2)





    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
client.close()
