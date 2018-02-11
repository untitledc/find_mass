# -*- coding: utf-8 -*-

import argparse
import os

from numpy.random import randint
from PIL import Image

GEN_IMAGE_ROOT = os.path.join('data', 'gen')


class ImageGenerator:
    def __init__(self, gen_width, gen_height, object_image_fn,
                 bg_color=(0, 0, 0)):
        self.gen_width = gen_width
        self.gen_height = gen_height
        self.obj_image = Image.open(object_image_fn)
        self.obj_width, self.obj_height = self.obj_image.size
        self.bg_color = bg_color
        self.fn2mass_center = {}

    def generate(self, output_fn):
        img = Image.new('RGB', (self.gen_width, self.gen_height),
                        color=self.bg_color)
        paste_x = randint(0, self.gen_width-self.obj_width)
        paste_y = randint(0, self.gen_height-self.obj_height)
        img.paste(self.obj_image, (paste_x, paste_y))
        img.save(output_fn)
        self.fn2mass_center[output_fn] = (paste_x + int(self.obj_width/2),
                                          paste_y + int(self.obj_height/2))

    def write_mass_center(self, output_fn):
        with open(output_fn, 'w') as out_f:
            for img_fn, mc in sorted(self.fn2mass_center.items()):
                out_f.write('{},{},{}\n'.format(img_fn, mc[0], mc[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Randomly generate image data')
    parser.add_argument('-n', '--amount', type=int, required=True,
                        help='number of images to generate')
    parser.add_argument('--obj_image', type=str, required=True,
                        help='filename of the object image')
    parser.add_argument('--height', type=int, default=320,
                        help='height of generated images')
    parser.add_argument('--width', type=int, default=320,
                        help='width of generated images')

    return parser.parse_args()


def main(args):
    gen_num = args.amount

    img_generator = ImageGenerator(args.width, args.height, args.obj_image)
    #img_generator = ImageGenerator(args.width, args.height, args.obj_image,
    #                               bg_color=(192,128,64))
    fn_len = len(str(gen_num))
    if not os.path.exists(GEN_IMAGE_ROOT):
        os.mkdir(GEN_IMAGE_ROOT)

    for i in range(gen_num):
        fn_prefix = str(i).zfill(fn_len)
        fn = os.path.join(GEN_IMAGE_ROOT, '{}.png'.format(fn_prefix))
        img_generator.generate(fn)

    img_generator.write_mass_center(os.path.join('data', 'answer.csv'))


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
