import time
import mmap
import pyttsx3
from pocketsphinx import LiveSpeech



VOICE_GUIDE_MODE = True
OCR_CAPTURE = False
TRANSLATE_CAPTURE = False

AUDIO_BUFFER = []


def voice_input_process():
    print(":::: voice_input_process run")

    speech = LiveSpeech()

    for phrase in speech:
        print(phrase)
        voice_string = str(phrase)

        symbol = list(voice_string.split())
        enable_mode(symbol)

    return
        


def enable_mode(symbol):
    global AUDIO_BUFFER, VOICE_GUIDE_MODE, OCR_CAPTURE, TRANSLATE_CAPTURE

    if 'one' in symbol:
        VOICE_GUIDE_MODE = not VOICE_GUIDE_MODE
        print('VOICE_GUIDE_MODE', VOICE_GUIDE_MODE)
        AUDIO_BUFFER.append("VOICE_GUIDE_MODE ON")

    elif 'two' in symbol:
        OCR_CAPTURE = not OCR_CAPTURE
        print('OCR_CAPTURE', OCR_CAPTURE)
        AUDIO_BUFFER.append("OCR_CAPTURE ON")

    elif 'three' in symbol:
        TRANSLATE_CAPTURE = not TRANSLATE_CAPTURE
        print('TRANSLATE_CAPTURE', TRANSLATE_CAPTURE)
        AUDIO_BUFFER.append("TRANSLATE_CAPTURE ON")

    with open("config.txt", "r+") as file:
        file.truncate(0)
        file.write(str(VOICE_GUIDE_MODE)+'\n')
        file.write(str(OCR_CAPTURE)+'\n')
        file.write(str(TRANSLATE_CAPTURE)+'\n')

    return



def audio_play_process():
    global AUDIO_BUFFER
    print(":::: audio_play_process run")
    open("buffer.txt", 'w').close()

    while True:
        print(AUDIO_BUFFER)
        if len(AUDIO_BUFFER) > 0:
            text = AUDIO_BUFFER.pop(0)
            tts_play(text)
            continue
        with open("buffer.txt", "r+") as file:
            '''
            mmapped_file = mmap.mmap(file.fileno(), 0)

            first_line = mmapped_file.readline()
            first_line = first_line.decode('utf-8')

            mmapped_file.seek(0)
            mmapped_file.write(mmapped_file[first_line.end():])

            mmapped_file.truncate()

            mmapped_file.close()
            '''
            while True:
                line = file.readline()
                if len(line.strip()) > 0:
                    AUDIO_BUFFER.append(line)
                if not line: 
                    break
            file.truncate(0)

        time.sleep(1)

    return



def tts_play(text, wait_time=0):
    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()

    time.sleep(wait_time)

    return


with open("config.txt", "r+") as file:
    file.truncate(0)
    file.write(str(VOICE_GUIDE_MODE)+'\n')
    file.write(str(OCR_CAPTURE)+'\n')
    file.write(str(TRANSLATE_CAPTURE)+'\n')