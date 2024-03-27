import math
import cv2
import numpy as np
import pytesseract

import argos_translate



def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



def find_sentence(ocr_data: dict, threshold:int=50) -> dict:
    result = dict()
    result['text'] = []
    result['left'] = []
    result['top'] = []
    result['width'] = []
    result['height'] = []

    sentence_string = ''
    sentence_left = -1
    sentence_top = -1
    sentence_width = -1
    sentence_height = -1

    for i in range(len(ocr_data['text'])):
        lv = ocr_data['level'][i]
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        conf = int(ocr_data['conf'][i])
        text = ocr_data['text'][i]
        text = text.strip()

        if lv == 4:
            if len(sentence_string.strip()) > 1:
                result['text'].append(sentence_string.strip())
                result['left'].append(sentence_left)
                result['top'].append(sentence_top)
                result['width'].append(sentence_width)
                result['height'].append(sentence_height)

                sentence_string = ''
                sentence_left = -1
                sentence_top = -1
                sentence_width = -1
                sentence_height = -1

        elif lv == 5:
            if conf > threshold and len(text) > 0:
                if sentence_left != -1 and sentence_left+sentence_width+w < x:
                    result['text'].append(sentence_string.strip())
                    result['left'].append(sentence_left)
                    result['top'].append(sentence_top)
                    result['width'].append(sentence_width)
                    result['height'].append(sentence_height)

                    sentence_string = ''
                    sentence_left = -1
                    sentence_top = -1
                    sentence_width = -1
                    sentence_height = -1

                    sentence_string += ' ' + text
                    sentence_left = x if sentence_left==-1 else min(sentence_left, x)
                    sentence_top = y if sentence_top==-1 else min(sentence_top, y)
                    sentence_width = w if sentence_width==-1 else max(sentence_left+sentence_width, x+w)-sentence_left
                    sentence_height = h if sentence_height==-1 else max(sentence_height, h)

                # Save words if they are consecutive
                else:
                    sentence_string += ' ' + text
                    sentence_left = x if sentence_left==-1 else min(sentence_left, x)
                    sentence_top = y if sentence_top==-1 else min(sentence_top, y)
                    sentence_width = w if sentence_width==-1 else max(sentence_left+sentence_width, x+w)-sentence_left
                    sentence_height = h if sentence_height==-1 else max(sentence_height, h)

    # Finally save the remaining value
    if len(sentence_string.strip()) > 1:
        result['text'].append(sentence_string.strip())
        result['left'].append(sentence_left)
        result['top'].append(sentence_top)
        result['width'].append(sentence_width)
        result['height'].append(sentence_height)

    return result



def find_paragraph(sentence_data: dict, threshold:float=1.5) -> dict:
    result = dict()
    result['text'] = []
    result['left'] = []
    result['top'] = []
    result['width'] = []
    result['height'] = []
    result['line'] = []
    result['lpos'] = []

    if len(sentence_data['text']) < 1:
        return result
    
    block_string = sentence_data['text'][0]
    block_left = sentence_data['left'][0]
    block_top = sentence_data['top'][0]
    block_width = sentence_data['width'][0]
    block_height = sentence_data['height'][0]
    line = 1
    line_pos = [(block_left, block_top, block_width, block_height)]
    base_height = block_height

    for i in range(1, len(sentence_data['text'])):
        text = sentence_data['text'][i]
        x = sentence_data['left'][i]
        y = sentence_data['top'][i]
        w = sentence_data['width'][i]
        h = sentence_data['height'][i]

        distance = calculate_distance(block_left, block_top+block_height, x,y)

        # Construct a paragraph based on the distance between the positions of the starting words of the two sentences
        if h*threshold < base_height or distance > base_height*threshold:
            result['text'].append(block_string)
            result['left'].append(block_left)
            result['top'].append(block_top)
            result['width'].append(block_width)
            result['height'].append(block_height)
            result['line'].append(line)
            result['lpos'].append(line_pos)

            block_string = text
            block_left = x
            block_top = y
            block_width = w
            block_height = h
            line = 1
            line_pos = [(block_left, block_top, block_width, block_height)]
            base_height = h

        else:
            block_string += ' ' + text
            block_left = min(block_left, x)
            block_width = max(block_left+block_width, x+w)-block_left
            block_height = max(block_top+block_height, y+h)-block_top
            line += 1
            line_pos.append((x, y, w, h))

    # Finally save the remaining value
    result['text'].append(block_string)
    result['left'].append(block_left)
    result['top'].append(block_top)
    result['width'].append(block_width)
    result['height'].append(block_height)
    result['line'].append(line)
    result['lpos'].append(line_pos)
    
    return result


def make_script(script_list):
    script = ''
    for text in script_list:
        script += text + '. '
    return script



def ocr_process(image, translate=False):
    lang = 'kor'
    if translate:
        lang = 'eng'
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=lang)

    sentence_data = find_sentence(ocr_data)

    paragraph_data = find_paragraph(sentence_data)

    script_text = []
    for i in range(len(paragraph_data['text'])):
        script_text.append(paragraph_data['text'][i])

    if translate:
        translated_script_text = []
        for i in script_text:
            translated_script_text.append(argos_translate.text_translate(text=i, dest='ko', src='en'))

        script = make_script(translated_script_text)

    else:
        script = make_script(script_text)
    
    with open("buffer.txt", "a") as file:
        file.write(script)

    return

