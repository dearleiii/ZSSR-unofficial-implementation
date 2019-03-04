import argparse
import PIL import Image
import numpy as np

def train(model, img, sr_factor, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_batches', type=int, default=15000, help='Number of batches to run')
    parser.add_argument('--crop', type = int, default = 128, help = 'Random crop size')
    parser.add_argument('--lr', type = float, default = 0.00001, help = 'Base learning rate for adam')
    parser.add_argument('factor', type = int, default = 2, help = 'Interpolation factor')
    parser.add_argument('--img', type=str, help = 'Path to input image')

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
        sys.exit(1)

    # Weight initialization
    model.apply(weights_init_kaiming)

    train(model, img, args.factor, args.num_batches, args.lr, args.crop)
    test(model, img, args.fator)
