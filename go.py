#!/usr/bin/env python3.7
import argparse
import re
import cv2
import glob
import numpy as np
from PIL import ImageFont, ImageDraw, Image  

Helvetica = "fonts/normal/Helvetica.ttc"
FONT_TYPES = glob.glob("fonts/crazy/*")

def parse_cli():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_image', type=str, help='image to put watermark on')
    parser.add_argument('output_image', type=str, help='output path')
    parser.add_argument('small_text', type=str, help='small watermark text', default=
                        'small watermark text')
    parser.add_argument('large_text', type=str, help='large watermark text', default=
                        'large watermark text')
    args = parser.parse_args()
    return args

def not_randi(x, y=None):
    return (max(0, x-1) + y) // 2 if y else max(0, x-1)

def put_watermark(img, small_text, large_text, spacing=30, crazy=True):
    if crazy:
        from numpy.random import randint as randi
    else:
        randi = not_randi

    space_locations = [m.start() for m in re.finditer(' ', small_text)]

    # Pass the image to PIL
    pil_im = Image.fromarray(img)

    txt_canvas = Image.new('RGBA', pil_im.size, (255,255,255, 0))
    draw = ImageDraw.Draw(txt_canvas)

    for i, y in enumerate(range(0, img.shape[0], spacing)):
        # use a truetype font
        font_size = randi(20, 21)
        font_type = FONT_TYPES[randi(len(FONT_TYPES))]
        font = ImageFont.truetype(font_type, font_size)

        x = -randi(10, 80)
        y += randi(3, 7)
        alpha = randi(40, 60)
        near_white = (255 - randi(30), 255 - randi(30), 255 - randi(30))

        insert_pos = space_locations[randi(len(space_locations))]
        augmented_text = small_text[:insert_pos] + (" " * randi(5)) + small_text[insert_pos:]
        draw.text((x, y), augmented_text, font=font, fill=near_white + (alpha,))

    large_font = ImageFont.truetype(Helvetica, 110)
    draw.text((55, 95), large_text, font=large_font, fill=(255, 255, 255, 100))

    # Get back the image to OpenCV
    watermarked = Image.alpha_composite(pil_im, txt_canvas)

    return np.array(watermarked)

def main():
    args = parse_cli()

    img = cv2.imread(args.input_image)
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    watermarked = put_watermark(img, args.small_text * 2, args.large_text)
    cv2.imwrite(args.output_image, watermarked)

if __name__ == "__main__":
    main()
