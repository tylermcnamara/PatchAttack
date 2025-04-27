import os
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path


# PatchAttack config
PA_cfg = edict() 

# config PatchAttack
def configure_PA(
        t_name,     # texture dictionary dir
        t_labels,   # all the labels in Dict, start from 0 and continuous
        target=False,   # type of attack
        area_occlu=0.035,   # area budget per patch (TPA)
        n_occlu=1, rl_batch=500, steps=50,  # RL hyper parameters
        HPA_bs=1,   # batch size for HPA
        MPA_color=False,    # set to True for MPA_RGB variant
        TPA_n_agents=10,    # How many sequential agents ran in TPA
       ):

    # Dictionry's shared params
    PA_cfg.t_name = t_name      # root folder name
    PA_cfg.t_labels = t_labels  # classes that dictionary will have textures for

    # Texture dictionary
    PA_cfg.conv = 5     # number of VGG blocks whose Gram matrices are kept
    PA_cfg.style_layer_choice = [1, 6, 11, 20, 29][:PA_cfg.conv]    # layer IDs (post ReLU)
    PA_cfg.style_channel_dims = [64, 128, 256, 512, 512][:PA_cfg.conv]  # channels per chosen layer
    PA_cfg.cam_thred = 0.8  # grad-cam mask threshold
    PA_cfg.n_clusters = 30  # buckets per class
    PA_cfg.cls_w = 0    # weight of class term while making textures
    PA_cfg.scale = 1    # spatial tiling scalar for textures
    PA_cfg.iter_num = 9999  # newest iteration to load or save

    # AdvPatch dictionary
    PA_cfg.image_shape = (3, 224, 224)  # sets valid area for patch placement
    PA_cfg.scale_min = 0.9  # affine range during training
    PA_cfg.scale_max = 1.1
    PA_cfg.rotate_max = 22.5
    PA_cfg.rotate_min = -22.5
    PA_cfg.batch_size = 16  # mini-batch size when optimizing a patch
    PA_cfg.percentage = 0.09    # % of image patch should take up
    PA_cfg.AP_lr = 10.0     # patch tensor learning rate
    PA_cfg.iterations = 500 # training iterations per patch


    # Attack's shared params
    PA_cfg.target = target  # targeted or non-targeted attack
    PA_cfg.n_occlu = n_occlu  # num of patches each agent can put on (default: 1), also used in 'HPA', 'MPA'
    PA_cfg.lr = 0.03  # learning rate for RL agent (default: 0.03)
    PA_cfg.rl_batch = rl_batch  # batch number when optimizing a RL agent (default: 500)
    PA_cfg.steps = steps  # steps to optimize each RL agent (default: 50), also used in HPA
    PA_cfg.sigma = 400  # sigam to control the area in HPA and MPA (default: 400.)
    PA_cfg.sigma_sched = []  # sigma schedule for the multiple occlusions (default: n-occlu * sigma), HPA and MPA
    if PA_cfg.sigma_sched == []:
        PA_cfg.sigma_sched = [PA_cfg.sigma]*PA_cfg.n_occlu

    # MPA
    PA_cfg.color = MPA_color  # flag to use MPA_RGB
    PA_cfg.critic = False  # actor-critic mode for each agent
    PA_cfg.dist_area = False  # use distributed-area mode
    PA_cfg.baseline_sub = True  # use baseline subtraction mode

    # TPA
    PA_cfg.n_agents = TPA_n_agents
    PA_cfg.f_noise = False  # filter the textures to be correctly classified by the model to fool, (default: False)
    PA_cfg.es_bnd = 1e-4  # early stop bound (default: 1e-4)
    PA_cfg.area_occlu = area_occlu  # occlusion area for each single patch (default: 0.04)
    PA_cfg.area_sched = []  # area schedule for the multiple agents (default: n-agents * area-occlu)
    if PA_cfg.area_sched == []:
        PA_cfg.area_sched = [PA_cfg.area_occlu] * PA_cfg.n_agents

    # HPA
    PA_cfg.HPA_bs = HPA_bs  # batch size for HPA (default: 1)
    # when bs is larger than 1, it means attacking mulitple images simultaneously. Othereise it is too slow.


    # Texture dict dirs (three level hierarchy)
    texture_dirs = []
    texture_sub_dirs = []
    texture_template_dirs = []

    for t_label in PA_cfg.t_labels:
        #   level 1 - per class root
        texture_dir = os.path.join(
            PA_cfg.t_name,
            'attention-style_t-label_{}'.format(t_label)
        )
        #   level 2 - conv depth, cam threshold and cluster count
        texture_sub_dir = os.path.join(
            texture_dir,
            'conv_{}_cam-thred_{}_n-clusters_{}'.format(
                PA_cfg.conv, PA_cfg.cam_thred, PA_cfg.n_clusters
            )
        )
        #   level 3 - class loss weight and tiling scale
        texture_template_dir = os.path.join(
            texture_sub_dir,
            'cls-w_{}_scale_{}'.format(
                PA_cfg.cls_w, PA_cfg.scale,
            )
        )
        texture_dirs.append(texture_dir)
        texture_sub_dirs.append(texture_sub_dir)
        texture_template_dirs.append(texture_template_dir)
        
    PA_cfg.texture_dirs = texture_dirs
    PA_cfg.texture_sub_dirs = texture_sub_dirs
    PA_cfg.texture_template_dirs = texture_template_dirs

    # AdvPatch dict dirs (one per label)
    PA_cfg.AdvPatch_dirs = []
    for t_label in PA_cfg.t_labels:
        AdvPatch_dir = os.path.join(
            PA_cfg.t_name,
            't-label_{}'.format(
                t_label
            ),
            'percentage_{}'.format(
                PA_cfg.percentage,
            ),
            'scale_{}-{}_rotate_{}-{}'.format(
                PA_cfg.scale_min, PA_cfg.scale_max,
                PA_cfg.rotate_min, PA_cfg.rotate_max,
            ),
            'LR_{}_batch_size_{}_iterations_{}'.format(
                PA_cfg.AP_lr, PA_cfg.batch_size, PA_cfg.iterations
            ),
        )
        PA_cfg.AdvPatch_dirs.append(AdvPatch_dir)

    # TPA attack results dirs (one per agent)
    TPA_attack_dirs = []
    for agent_index in range(PA_cfg.n_agents):
        attack_dir = os.path.join(
            'target' if PA_cfg.target else 'non-target',
            f'c{PA_cfg.conv}_th{int(PA_cfg.cam_thred * 10)}_cl{PA_cfg.n_clusters}',
            f'o{PA_cfg.n_occlu}_fn{int(PA_cfg.f_noise)}_lr{int(PA_cfg.lr * 100)}_rb{PA_cfg.rl_batch}_s{PA_cfg.steps}',
            f'as{agent_index + 1}',
        )
        TPA_attack_dirs.append(attack_dir)
    PA_cfg.TPA_attack_dirs = TPA_attack_dirs


    # MPA attack results dirs
    MPA_attack_dir = os.path.join(
        "target" if PA_cfg.target else "non-target",
            f"oc{PA_cfg.n_occlu}"
            f"_clr{int(PA_cfg.color)}"
            f"_lr{int(PA_cfg.lr * 100)}"
            f"_crt{int(PA_cfg.critic)}"
            f"_rb{PA_cfg.rl_batch}"
            f"_s{PA_cfg.steps}"
        ,
        "sigma-" + "-".join(str(s) for s in PA_cfg.sigma_sched),
    )
    PA_cfg.MPA_attack_dir = MPA_attack_dir


    # HPA attack results dirs
    HPA_attack_dir = os.path.join(
        "target" if PA_cfg.target else "non-target",
        f"oc{PA_cfg.n_occlu}_s{PA_cfg.steps}",
        "sigma-" + "-".join(str(s) for s in PA_cfg.sigma_sched),
    )
    PA_cfg.HPA_attack_dir = HPA_attack_dir


    # AP attack dirs
    temp_dir = os.path.join(*Path(PA_cfg.AdvPatch_dirs[0]).parts[2:])
    AP_attack_dir = os.path.join(
        "target" if PA_cfg.target else "non-target",
        temp_dir,
    )
    PA_cfg.AP_attack_dir = AP_attack_dir
