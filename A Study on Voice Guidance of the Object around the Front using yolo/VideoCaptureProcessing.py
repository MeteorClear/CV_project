import cv2
import numpy as np
import time
import mmap

import OCRProcessing



POSITION_MATRIX = [[set(), set(), set()],
                   [set(), set(), set()],
                   [set(), set(), set()]]

CHANGES_MATRIX = [[set(), set(), set()],
                  [set(), set(), set()],
                  [set(), set(), set()]]


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"]


POSITION_NAMES = [['left top', 'top', 'right top'],
                  ['left', 'front', 'right'],
                  ['left bottom', 'bottom', 'right bottom']]


CONFIG_MODE = []


def find_midpoint(x, y, w, h):
    return (x+w//2, y+h//2)


def find_direction(width, height, posx, posy):
    ptx = width//3
    pty = height//3

    dx = -1
    dy = -1

    if 0 <= posx <= ptx:
        dx = 0
    elif ptx < posx < ptx*2:
        dx = 1
    else:
        dx = 2
    
    if 0 <= posy <= pty:
        dy = 0
    elif pty < posy < pty*2:
        dy = 1
    else:
        dy = 2

    return (dx, dy)


def find_object_position():
    global CHANGES_MATRIX, POSITION_MATRIX

    changes_log = ''

    for i in range(3):
        for j in range(3):
            change = CHANGES_MATRIX[i][j].difference(POSITION_MATRIX[i][j])

            if len(change) < 1:
                continue

            changes_log += str(*change) + ' ' + POSITION_NAMES[i][j] + '. '

            #POSITION_MATRIX[i][j] = CHANGES_MATRIX[i][j].copy()
    POSITION_MATRIX = CHANGES_MATRIX[:]

    CHANGES_MATRIX = [[set(), set(), set()],
                      [set(), set(), set()],
                      [set(), set(), set()]]
    
    return changes_log



def video_capture_process(div_num=0, width=416, height=416, frame_count=10, detect_confidence=0.5):
    global CONFIG_MODE, CHANGES_MATRIX
    print(":::: video_capture_process run")
    fps = []
    frame_num = 0


    # load yolo weights, configs, layers
    net = cv2.dnn.readNet('.\\yolo_data\\yolov3-tiny.weights', '.\\yolo_data\\yolov3-tiny.cfg')
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()
    if isinstance(output_layer_indices, int):
        output_layer_indices = [output_layer_indices]
    output_layers = [layer_names[i - 1] for i in output_layer_indices]


    # set for opencv video capture
    capture = cv2.VideoCapture(div_num)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    with open("config.txt", "r+") as file:
        CONFIG_MODE = []
        while True:
            line = file.readline()
            if len(line.strip()) > 0:
                CONFIG_MODE.append(line.strip())
            if not line: 
                break


    # video capture
    while capture.isOpened():
        success, image = capture.read()
        res = image.copy()
        image_height, image_width, _c = image.shape

        if not success:
            continue


        # fps calculate
        frame_num += 1
        fps.append(time.time())
        if len(fps) > frame_count:
            fps.pop(0)
            cv2.putText(res, str(int(frame_count/(fps[9] - fps[0]))), (25,30), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)


        # load network, blob object
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)


        # organize detected data
        boxes = []
        class_ids = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > detect_confidence:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)

                    d_w = int(detection[2] * image_width)
                    d_h = int(detection[3] * image_height)
                    d_x = int(center_x - d_w / 2)
                    d_y = int(center_y - d_h / 2)

                    boxes.append([d_x, d_y, d_w, d_h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


        # draw object
        
        cv2.circle(res, (image_width//2, image_height//2), 30, (0,0,255), 3)
        for i in range(len(boxes)):
            if i in indexes:
                r_x, r_y, r_w, r_h = boxes[i]
                class_id_num = class_ids[i]
                class_id_name = COCO_CLASSES[class_id_num]
                m_x, m_y = find_midpoint(r_x, r_y, r_w, r_h)

                px, py = find_direction(image_width, image_height, m_x, m_y)
                if frame_num % 100 == 0:
                    CHANGES_MATRIX[py][px].add(class_id_name)

                cv2.rectangle(res, (r_x, r_y), (r_x + r_w, r_y + r_h), (0,255,0), 2)
                cv2.putText(res, class_id_name, (r_x, r_y-5), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)
                cv2.line(res, (image_width//2, image_height//2), (m_x, m_y), (255,0,0), 1)
                cv2.circle(res, (m_x, m_y), 5, (0,255,0), -1)


        if len(CONFIG_MODE) > 1 and CONFIG_MODE[1]=='True':
            OCRProcessing.ocr_process(image=image, translate=bool(CONFIG_MODE[2]))
        
        
        
        if frame_num % 50 == 0:
            # find object position
            if len(CONFIG_MODE) > 1 and CONFIG_MODE[0]=='True':
                changes_log = find_object_position()
                if len(changes_log.strip()) > 1:
                    #AudioTextProcessing.AUDIO_BUFFER.append(changes_log)
                    print(changes_log)
                    with open("buffer.txt", "a") as file:
                        '''
                        mmapped_file = mmap.mmap(file.fileno(), 0)
                        mmapped_file.write(changes_log.encode('utf-8'))
                        mmapped_file.close()
                        '''
                        file.write(changes_log)

            with open("config.txt", "r+") as file:
                CONFIG_MODE = []
                while True:
                    line = file.readline()
                    if len(line.strip()) > 0:
                        CONFIG_MODE.append(line.strip())
                    if not line: 
                        break



        cv2.imshow('result', res)

        key_input = cv2.waitKey(1)

        if key_input & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    return

