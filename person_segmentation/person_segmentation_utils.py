import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from enum import Enum

def get_all_segmentation_map(rgb_image_path, image_size=200, device='cpu'):
    model = torch.hub.load('pytorch/vision:v0.13.0', 'deeplabv3_resnet50', weights='COCO_WITH_VOC_LABELS_V1')

    model.eval()

    input_image = Image.open(rgb_image_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():
        '''The output here is of shape (21, H, W), and at each location, there are unnormalized probabilities 
        corresponding to the prediction of each class. To get the maximum prediction of each class, and then use it 
        for a downstream task, you can do output_predictions = output.argmax(0).'''
        output = model(input_batch)['out'][0]

    classes = Enum('segmentation_class_idx', ['background',
                                              'aeroplane',
                                              'bicycle',
                                              'bird',
                                              'boat',
                                              'bottle',
                                              'bus',
                                              'car',
                                              'cat',
                                              'chair',
                                              'cow',
                                              'dining_table',
                                              'dog',
                                              'horse',
                                              'motorbike',
                                              'person',
                                              'potted_plant',
                                              'sheep',
                                              'sofa',
                                              'train',
                                              'tv_monitor',
                                              'num_classes'
                                              ], start=0)

    return input_image, output, classes


def show_combined_segmentation_map(rgb_image_path, image_size=200, device='cpu'):
    input_images_and_output_predictions = get_all_segmentation_map(rgb_image_path, image_size, device)
    input_image = input_images_and_output_predictions[0]
    output = input_images_and_output_predictions[1]  # output.shape = (21, H, W) # 21 output classes in the image
    segmentation_class_idx = input_images_and_output_predictions[-1]

    # Get the class of each pixel
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(segmentation_class_idx.num_classes.value)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    plt.imshow(r)
    plt.show()


def get_human_segmentation_map(rgb_image_path, image_size=100, device='cpu'):
    input_images_and_output_predictions = get_all_segmentation_map(rgb_image_path, image_size, device)
    input_image = input_images_and_output_predictions[0]
    output = input_images_and_output_predictions[1].to(torch.float)  # (21, H, W)
    segmentation_class_idx = input_images_and_output_predictions[-1]

    print(output.shape)

    # Get the class of each pixel
    output_predictions = output.argmax(0)  # (H,W) # Each value in this matrix is the class

    print(segmentation_class_idx.person.value)

    human_mask_tensor = (output_predictions == segmentation_class_idx.person.value)
    human_mask_numpy = human_mask_tensor.cpu().numpy()

    return human_mask_numpy, human_mask_tensor


def show_human_segmentation_map(rgb_image_path, image_size=100, device='cpu'):
    human_mask_numpy = get_human_segmentation_map(rgb_image_path, image_size, device)[0]

    plt.imshow(human_mask_numpy)
    plt.show()

    return human_mask_numpy
