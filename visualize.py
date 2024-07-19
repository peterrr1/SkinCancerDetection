from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as v2

def show_images(img_1, img_2):

    fig = plt.figure(figsize=(10, 10))
    rows, cols = 1, 2
    fig.add_subplot(rows, cols, 1)
    plt.imshow(v2.ToPILImage()(img_1))
    fig.add_subplot(rows, cols, 2)
    plt.imshow(v2.ToPILImage()(img_2), cmap=plt.cm.gray)
    plt.show()


def create_histogram(img) -> list:
    ## Iterate over grayscale image with 3 channels
    pixels = [0] * 256
    img = img[0]
    
    for iy, ix in np.ndindex(img.shape):
        value = img[iy, ix]
        pixels[value] += 1
    return pixels

def plot_histogram(data) -> None:
    plt.bar(x = list(range(0, 256, 1)), height = data)
    plt.title('Histogram')
    
    plt.show()