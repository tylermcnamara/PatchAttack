import os
import torch
import torchvision
import torchvision.models as Models
from torchvision import transforms

import PatchAttack.utils as utils
from PatchAttack.PatchAttack_config import PA_cfg
from PatchAttack.TextureDict_extractor import vgg19_extractor as texture_generator

torch_cuda = 0


# custom data agent for texture generation
class custom_data_agent():

    def __init__(self, train_dataset, labels_mapping=None):
        self.train_dataset = train_dataset
        self.labels_mapping = labels_mapping

    def get_indices(self, label, save_dir, correct=False, cnn=None, train=True):
        '''
        input:
        label: int
        correct: flag to return the indices of the data point which is crrectly classified by the cnn
        cnn: pytorch model
        process_PIL: transform used in the 'correct' mode
        return:
        torch.tensor containing the indices in self.train_dataset or self.val_dataset, 
        or custom dataset when in 'correct' mode
        '''

        # 'cache' file to avoid duplicate results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, 'label_{}_train-set_{}_correct_{}.pt'.format(label, train, correct))
        # use cached tensor
        if os.path.exists(file_name):
            indices = torch.load(file_name) # load saved indices
            return indices                  # early exit
        else: # compute indices from scratch
            if train:
                targets_tensor = torch.Tensor(self.train_dataset.targets)
            else:
                raise NotImplementedError
            # boolean mask that keeps samples whose ground-truth label matches input label
            temp = torch.arange(len(targets_tensor))
            indices = temp[targets_tensor==label]

            # optional filter to only keep correctly identified classes by the target model
            if correct:
                cnn = cnn.cuda(torch_cuda).eval()   # inference mode on GPU
                with torch.no_grad():   # no gradients needed
                    wrong_set = []      # gather misclassified labels
                    label_tensor = torch.Tensor([label]).long().cuda(torch_cuda)
                    for index in indices:
                        #   pull image exactly as the dataset returns it
                        input_tensor = self.train_dataset.__getitem__(index)[0]
                        input_tensor = input_tensor.cuda(torch_cuda).unsqueeze(0)
                        # forward pass through reference network
                        output_tensor = cnn(input_tensor)
                        pred = output_tensor.argmax()

                        # handle merged class experiments with labels mapping
                        if self.labels_mapping is not None:
                            valid_pred = False
                            for i_mapping in range(len(self.labels_mapping)):
                                if pred in self.labels_mapping[i_mapping]:
                                    pred = torch.LongTensor([i_mapping])
                                    valid_pred = True
                                    break
                            if not valid_pred:
                                pred = torch.LongTensor([-1])
                            pred = pred.cuda(torch_cuda)
                         #  mark samples the cnn got wrong for removal
                        if pred != label_tensor:
                            wrong_set.append(index)
                    #   drop every misclassified index
                    for item in wrong_set:
                        indices = indices[indices!=item]
            # save for future runs
            torch.save(indices, file_name)
            return indices


def build(DA, t_labels):
    """
    Generate Gram matrix based texture templates for each label in t_labels

    Parameters:
         DA = custom_data_agent
            - provides fast access to training indices

        t_labels = list of indices
            - class ids to build textures for the dictionary
    """
    # MAIN variables
    labels = torch.LongTensor(t_labels).cuda(torch_cuda)
    
    # set texture_templates
    texture_templates = []
    
    # configure texture_generator
    texture_generator.attention_threshold = PA_cfg.cam_thred    # grad-CAM mask threshold
    texture_generator.style_choice = PA_cfg.style_layer_choice  # which VGG layers form the style vector
    
    # prepare vgg19 (reference classifier) to extract textures
    cnn = Models.vgg19(pretrained=True).cuda(torch_cuda).eval()
    
    # build texture templates for each t_label
    for texture_index in range(len(PA_cfg.texture_template_dirs)):
        # set noises
        noises = []
        
        # check whether the dictionary has been built
        if os.path.exists(
            os.path.join(
                PA_cfg.texture_template_dirs[texture_index], 
                'cluster_{}'.format(PA_cfg.n_clusters-1), 
                'iter_{}.pt'.format(PA_cfg.iter_num)
                )
            ):
            print('texture dictionary of label_{} is already built, turn to next label...'.format(
                t_labels[texture_index]))
        else:

            # get a clean set of image indices for this label
            indices = DA.get_indices(
                t_labels[texture_index], save_dir=PA_cfg.texture_dirs[texture_index],
                cnn=cnn, correct=True
            )

            # cluster their grad-CAM weighted style vectors into buckets
            kmeans = texture_generator.get_kmeans_style(
                indices, DA.train_dataset,
                save_dir=PA_cfg.texture_sub_dirs[texture_index],
                n_clusters=PA_cfg.n_clusters,
            )
            
            # flatten/inverse back to per layer gram matrices for generator
            target_clusters = texture_generator.flatten_style(
                torch.from_numpy(kmeans.cluster_centers_).cuda(torch_cuda), 
                inv=True, 
                dims=PA_cfg.style_channel_dims
            )
            
            # generate textures
            for c_index in range(PA_cfg.n_clusters):
                noise = texture_generator.generate_image_from_style(
                    target_clusters[c_index],
                    os.path.join(
                        PA_cfg.texture_template_dirs[texture_index],
                        'cluster_{}'.format(c_index)
                    ),
                    label=labels[texture_index],    # classification label
                    cls_w=PA_cfg.cls_w,             # weight for classification terms
                    scale=PA_cfg.scale,             # spatial tiling factor
                )
                print('index: {} | label_{} | {}th of {} texture is generated'.format(
                    texture_index, t_labels[texture_index], c_index+1, PA_cfg.n_clusters))

                # update noises
                noises.append(noise)
                
        # print status
        print('index: {} | label_{} | {} textures has been prepared!'.format(
            texture_index, t_labels[texture_index], len(noises)))
            
        # update texture templates
        texture_templates.append(noises)
            
        # release memory
        torch.cuda.empty_cache()
            
    # print status
    texture_templates_status = [len(item) for item in texture_templates]
    print('{} different texture types | template number details: {}'.format(
        texture_templates_status, len(texture_templates_status)))
    
