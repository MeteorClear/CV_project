from multiprocessing import Process
import time
import cv2

import VideoCaptureProcessing
import AudioTextProcessing
import OCRProcessing


if __name__ == '__main__':
    process_audio_play = Process(target=AudioTextProcessing.audio_play_process)
    process_voice_input = Process(target=AudioTextProcessing.voice_input_process)
    process_video_capture = Process(target=VideoCaptureProcessing.video_capture_process)

    process_audio_play.start()
    process_voice_input.start()
    process_video_capture.start()
    
    count = 0
    while True:
        count = count + 1

        key_input = cv2.waitKey(10)

        if key_input & 0xFF == 27:
            break

    process_audio_play.terminate()
    process_voice_input.terminate()
    process_video_capture.terminate()