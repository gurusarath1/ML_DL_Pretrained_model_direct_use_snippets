import torch
import torchvision
import human_segmentation_utils


if __name__ == '__main__':

    print('Human segmentation running .....')

    human_segmentation_utils.show_combined_segmentation_map('./sample_images/2.jpg', 200, 'cuda')