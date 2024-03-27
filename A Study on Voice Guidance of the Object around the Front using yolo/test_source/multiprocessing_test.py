from multiprocessing import Process
import time

import cv2
import numpy as np
import pytesseract
from pocketsphinx import LiveSpeech


chk_mode_1 = False
chk_mode_2 = False
def voice_input_process():
    global chk_mode_1, chk_mode_2
    print("voice_input_process run")
    speech = LiveSpeech()
    for phrase in speech:
        print(phrase)
        voice_string = str(phrase)

        symbol = list(voice_string.split())
        if 'one' in symbol:
            chk_mode_1 = not chk_mode_1
            print('mode1', chk_mode_1)
        elif 'two' in symbol:
            chk_mode_2 = not chk_mode_2
            print('mode2', chk_mode_2)



def video_capture_process(div_num=0, width=416, height=416, frame_count=10):
    print("video_capture_process run")
    fps = []

    net = cv2.dnn.readNet('.\\yolo_data\\yolov3-tiny.weights', '.\\yolo_data\\yolov3-tiny.cfg')
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()
    if isinstance(output_layer_indices, int):
        output_layer_indices = [output_layer_indices]
    output_layers = [layer_names[i - 1] for i in output_layer_indices]

    capture = cv2.VideoCapture(div_num)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while capture.isOpened():
        success, image = capture.read()
        res = image.copy()
        height, width, channels = image.shape

        if not success:
            continue

        fps.append(time.time())
        if len(fps) > frame_count:
            fps.pop(0)
            cv2.putText(res, str(int(frame_count/(fps[9] - fps[0]))), (25,30), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        class_ids = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    d_w = int(detection[2] * width)
                    d_h = int(detection[3] * height)
                    d_x = int(center_x - d_w / 2)
                    d_y = int(center_y - d_h / 2)

                    boxes.append([d_x, d_y, d_w, d_h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                r_x, r_y, r_w, r_h = boxes[i]
                cv2.rectangle(res, (r_x, r_y), (r_x + r_w, r_y + r_h), (0,255,0), 2)
        
        cv2.imshow('result', res)

        key_input = cv2.waitKey(1)

        if key_input & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    p_a = Process(target=voice_input_process)
    p_b = Process(target=video_capture_process)
    count = 0
    while True:
        count = count + 1
        print("count: ", count)
        if count == 100000000:
            count = 0
        if count == 50:
            p_a.start()
        if count == 70:
            p_b.start()
        time.sleep(0.1)