from matplotlib import pyplot as plt
import torchvision.transforms as v2

def show_images(img_1, img_2):
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 1, 2
    fig.add_subplot(rows, cols, 1)
    plt.imshow(v2.ToPILImage()(img_1))
    fig.add_subplot(rows, cols, 2)
    plt.imshow(v2.ToPILImage()(img_2))
    plt.show()