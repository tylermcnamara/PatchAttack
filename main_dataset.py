import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.models as Models
import torchvision.transforms as Transforms
import torchvision.datasets as datasets
from torchvision.models import ResNet50_Weights
from PIL import Image
import urllib.request
import PatchAttack.PatchAttack_attackers as PA
from PatchAttack import utils
from PatchAttack.PatchAttack_config import configure_PA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def printPrediction(model, input_tensor, human_readable_labels):
    with torch.no_grad():
        pred = model(input_tensor.unsqueeze(0)).argmax(dim=1).item()
        print('pred: {}__{}'.format(pred, human_readable_labels[pred]))


def runTPA(dir_title, model, index, input_tensor, label_tensor, human_readable_labels):
    # configure PA_cfg
    configure_PA(
        t_name='TextureDict_demo',  # texture dictionary dir
        t_labels=np.arange(1000).tolist(),  # all the labels in Dict, start from 0 and continuous
        target=True,  # targeted or non-targeted attack
        area_occlu=0.035,  # area of each patch
        n_occlu=1,  # need to set to be 1 in TPA
        rl_batch=500, steps=50,
        TPA_n_agents=10  # maximum number of patches allowed in TPA
    )

    TPA = PA.TPA(dir_title)
    adv_image, rcd_list = TPA.attack(
        model=model,
        input_tensor=input_tensor,
        label_tensor=label_tensor,
        target=723,  # For non targeted attack, use textures of a randomly chosen class
        input_name='{}'.format(index),
    )

    combos = rcd_list[-1].combos[0]  # all the actions RL agent has selected
    area = TPA.calculate_area(adv_image, combos)
    print('Area used: {:.4f}'.format(area))

    utils.data_agent.show_image_from_tensor(adv_image, inv=True)

    printPrediction(model=model, input_tensor=adv_image, human_readable_labels=human_readable_labels)




def main():
    with open(os.path.join('./path_to_your_dataset_labels_folder')) as file:
        lines = file.readlines()
    human_readable_labels = [line.split("'")[1] for line in lines]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()  # use your custom model here

    preprocess = Transforms.Compose([
        Transforms.Resize(256),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        utils.data_agent.normalize,
    ])


    max_images = 1000   #   how many samples are attacked in the dataset
    dataset = datasets.ImageFolder(
        root='./path_to_your_dataset_folder',
        transform=preprocess
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for index, (input_tensor, label_tensor) in enumerate(data_loader):
        if index >= max_images:
            break
        input_tensor = input_tensor.squeeze(0).cuda()
        label_tensor = label_tensor.cuda()

        printPrediction(model=model, input_tensor=input_tensor, human_readable_labels=human_readable_labels)

        utils.data_agent.show_image_from_tensor(input_tensor, inv=True)

        dir_title = 'PatchAttack_tutorial_batch'  # new dir for batch attacks

        runTPA(
            dir_title=dir_title,
            index=index,
            model=model,
            input_tensor=input_tensor,
            label_tensor=label_tensor,
            human_readable_labels=human_readable_labels
        )


if __name__ == "__main__":
    sys.exit(int(main() or 0))