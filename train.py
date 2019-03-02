import argparse
import PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_batches', type=int, default=15000, help='Number of batches to run')
    parser.add_argument('--crop')
    parser.add_argument('--lr')
    parser.add_argument('factor')
    parser.add_argument('--img', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.img)  # does not open image until start processing data
    num_channels = len(np.array(img).shape)

    if num_channels == 3:
        model = ZSSRNet(input_channel = 3)
    elif num_channels == 1:
        model = ZSSRNet(input_channel = 1)
    else:
        print("Expecting RGB or gray input image, instead got ", img.size)
    