import os
import copy
import numpy as np
import torch
from PatchAttack.PatchAttack_config import PA_cfg
from PatchAttack.PatchAttack_agents import TPA_agent, MPA_agent, HPA_agent
from easydict import EasyDict as edict
import kornia
from kornia.geometry.transform import translate
import numpy.core.multiarray
# Register EasyDict as a “safe” global for torch.load so pickled rcd objects
# don’t raise an unpickling‑guard exception (torch >= 2.0 tightened defaults).
torch.serialization.add_safe_globals([
    edict,

])


torch_cuda = 0
# ===================================================================
# TPA - Textured Patch Attack (divide and conquer, sequential agents)
# ===================================================================
class TPA():

    def __init__(self, dir_title):
        #   create result directory paths for each agent
        dir_title = os.path.join(dir_title, PA_cfg.t_name)

        # attack dirs
        self.attack_dirs = [os.path.join(
            'PatchAttack_result',
            'TPA',
            dir_title,
            item,
        ) for item in PA_cfg.TPA_attack_dirs]

    # ---------------------------------------------------------------------------
    #  Main method - run (or resume) the sequential TPA agents on a single image
    # ---------------------------------------------------------------------------
    def attack(self, model, input_tensor, label_tensor, target, input_name='temp', target_in_dict_mapping=None):

        #   build or create work folders for each image
        attack_dirs = [os.path.join(item, input_name, 'texture_used_{}'.format(target)) for item in self.attack_dirs]
        for attack_dir in attack_dirs:
            if not os.path.exists(attack_dir):
                os.makedirs(attack_dir)

        # load texture images
        noises = []
        #   determine the dictionary index for the target layers
        if target_in_dict_mapping is None:
            target_in_dict = target
        else:
            #   map external target to internal dictionary index
            target_in_dict = target_in_dict_mapping[target][0]

        # check whether the dictionary has been built
        if os.path.exists(
                os.path.join(
                    PA_cfg.texture_template_dirs[target_in_dict],
                    'cluster_{}'.format(PA_cfg.n_clusters - 1),
                    'iter_{}.pt'.format(PA_cfg.iter_num)
                )
        ):
            print('texture dictionary of label_{} is already built, loading...'.format(
                target_in_dict))

            #   load each clusters texture patch
            for c_index in range(PA_cfg.n_clusters):
                noise_to_load = torch.load(os.path.join(
                    PA_cfg.texture_template_dirs[target_in_dict],
                    'cluster_{}'.format(c_index),
                    'iter_{}.pt'.format(PA_cfg.iter_num)
                ), weights_only=True)
                noises.append(noise_to_load)
        else:
            assert False, 'texture dictionary of label {} not found, please generate it'.format(target_in_dict)

        # print status, how many textured were loaded
        print('target_{} | {} texture images has been prepared!'.format(
            target_in_dict, len(noises)))

        # release memory
        torch.cuda.empty_cache()

        # filter textures that already cause misclassification
        if PA_cfg.f_noise:
            with torch.no_grad():
                textures_used = []
                for noise in noises:
                    #   repeat texture to match image size and move to GPU
                    input_noise = texture_generator.spatial_repeat(
                        noise.cuda(torch_cuda), PA_cfg.scale
                    )
                    #   predict label for the patched image
                    output = F.softmax(model(input_noise.unsqueeze(0)), dim=1)
                    #   only keep the textures that cause targeted misclassification
                    if output.argmax() == target:
                        textures_used.append(input_noise)
            print('after filtering, there are {} texture images to use'.format(len(textures_used)))
        else:
            textures_used = noises
            print('not filtering the texture images')

        # set up records for each agent
        t_rcd = edict()
        t_rcd.combos = []
        t_rcd.non_target_success = []
        t_rcd.target_success = []
        t_rcd.time_used = []
        t_rcd.queries = []

        #   duplicate record template for each agent
        rcd = [copy.deepcopy(t_rcd) for _ in range(PA_cfg.n_agents)]

        # check for any previously finished agents to continue
        n_pre_agents = 0
        for agent_index in range(PA_cfg.n_agents - 1, -1, -1):
            #   found last completed agent index
            if os.path.exists(os.path.join(
                    attack_dirs[agent_index],
                    'finished.pt',
            )):
                n_pre_agents = agent_index + 1
                #   load its record
                to_load = edict(torch.load(os.path.join(
                    attack_dirs[agent_index],
                    'rcd.pt',
                ), weights_only=True))
                rcd[agent_index] = copy.deepcopy(to_load)
                break
        load_rcd = n_pre_agents != 0

        #   prepare target tensor on GPU
        target_tensor = torch.LongTensor([target]).cuda(torch_cuda)
        #   launch the divide and conquer TPA agent sequence
        p_image, combos, non_target_success, target_success, time_used, queries = TPA_agent.DC(
            model=model,
            p_image=input_tensor,
            label=label_tensor,
            noises_used=textures_used,
            noises_label=target_tensor,
            area_sched=PA_cfg.area_sched,
            n_boxes=PA_cfg.n_agents,
            attack_type='target' if PA_cfg.target else 'non-target',
            num_occlu=PA_cfg.n_occlu,
            lr=PA_cfg.lr,
            rl_batch=PA_cfg.rl_batch,
            steps=PA_cfg.steps,
            n_pre_agents=n_pre_agents,
            to_load=to_load if load_rcd else None,
            load_index=0 if load_rcd else None,  # set it to 0 for single input tensor
        )

        #   update and save records for newly run agents
        for a_i in range(n_pre_agents, PA_cfg.n_agents):
            rcd[a_i].combos.append(combos[a_i])
            rcd[a_i].non_target_success.append(non_target_success[a_i])
            rcd[a_i].target_success.append(target_success[a_i])
            rcd[a_i].time_used.append(time_used[a_i])
            rcd[a_i].queries.append(queries[a_i])
            # save records
            torch.save(clean_for_saving(rcd[a_i]), os.path.join(attack_dirs[a_i], 'rcd.pt'))

        #   mark each agent as finished
        for agent_index in range(n_pre_agents, PA_cfg.n_agents):
            torch.save('finished!', os.path.join(
                attack_dirs[agent_index], 'finished.pt'
            ))
        #   return the adversarial image and all records
        return p_image, rcd

    @staticmethod
    def calculate_area(p_image, combos):
        # apply combos
        H, W = p_image.size()[-2:]
        #   pre-compute side lengths for each scheduled occlusion area
        pre_size_occlu = [torch.Tensor([H * W * item]).sqrt_().floor_().long().item()
                          for item in PA_cfg.area_sched]
        #   generate a combined mask from all combos
        p_mask = TPA_agent.combo_to_image(
            combo=torch.cat(combos, dim=1),
            num_occlu=len(combos),
            mask=None,
            image_batch=p_image.unsqueeze(0),
            noises=None,
            size_occlu=pre_size_occlu,
            output_p_masks=True
        ).squeeze(0)
        #   compute fraction of pixels occluded
        areas = p_mask.float().sum() / (H * W)

        return areas

# ===================================================================
# MPA - Monochrome Patch Attack (single agent, single color patch)
# ===================================================================
class MPA():
    
    def __init__(self, dir_title):
        #   create a result directory path
        self.attack_dir = os.path.join(
            'PatchAttack_result',
            'MPA',
            dir_title,
            PA_cfg.MPA_attack_dir,
        )

    # ---------------------------------------------------------------------------
    #  Main method - run (or resume) the single-shot MPA single image
    # ---------------------------------------------------------------------------
    def attack(self, model, input_tensor, label_tensor, target, input_name='temp'):
        #   build or create work folders for image
        attack_dir = os.path.join(self.attack_dir, input_name)
        if not os.path.exists(attack_dir):
            print("Directory {} doesn't exist. Creating new directory".format(input_name))
            os.makedirs(attack_dir)
        #   if a finished flag file exists, load its results
        if os.path.exists(os.path.join(attack_dir, 'finished.pt')):
            loaded = torch.load(os.path.join(attack_dir, 'rcd.pt'), weights_only=True)
            rcd = edict(loaded)

            
        else:
            
            # set records
            rcd = edict()
            rcd.masks = []              #   list of occlusion masks
            rcd.RGB_paintings = []      #   list of patch color placements
            rcd.combos = []             #   parameter combos for patches
            rcd.areas = []              #   occluded area fractions
            rcd.non_target_success = []
            rcd.target_success = []
            rcd.queries = []            #   model query counts
            rcd.time_used = []          #   runtime for each run

            #   prepare target tensor on GPU
            target_tensor = torch.LongTensor([target]).cuda(torch_cuda)
            #   launch the MPA agent
            mask, RGB_painting, combo, area, success, queries, time_used = MPA_agent.attack(
                model=model, 
                input_tensor=input_tensor, 
                target_tensor=label_tensor, 
                sigma=PA_cfg.sigma, 
                target_type=('random-fix-target', target_tensor) if PA_cfg.target else 'non-target', 
                lr=PA_cfg.lr, 
                distributed_area=PA_cfg.dist_area, 
                critic=PA_cfg.critic, 
                baseline_subtraction=PA_cfg.baseline_sub, 
                color=PA_cfg.color, 
                num_occlu=PA_cfg.n_occlu, 
                rl_batch=PA_cfg.rl_batch, 
                steps=PA_cfg.steps
            )

            #   add attack results to the record
            rcd.masks.append(mask)
            rcd.RGB_paintings.append(RGB_painting)
            rcd.combos.append(combo)
            rcd.areas.append(area)
            rcd.non_target_success.append(success[0])
            rcd.target_success.append(success[1])
            rcd.queries.append(queries)
            rcd.time_used.append(time_used)

            # save records
            torch.save(dict(rcd), os.path.join(attack_dir, 'rcd.pt'))

            # finished flag
            torch.save('finished!', os.path.join(attack_dir, 'finished.pt'))
        
        # print records
        print('*** '
              'non_target_success: {} | target_success: {} | '
              'queries: {:.4f} | occluded area: {:.4f} | '
              .format(
                  rcd.non_target_success,
                  rcd.target_success,
                  rcd.queries[0],
                  rcd.areas[0].item(),
              )
             )
        
        #   generate adv_image based on color preference
        load_index = 0
        if PA_cfg.color:
            masked_input_tensor = MPA_agent.paint(
                input_tensor.unsqueeze(0), 
                rcd.masks[load_index].unsqueeze(0), 
                rcd.RGB_paintings[load_index])
        else:
            masked_input_tensor = input_tensor.unsqueeze(0)*rcd.masks[load_index].unsqueeze(0).cuda(torch_cuda)
        #   return the adversarial image and the records
        return masked_input_tensor, rcd
    
    @staticmethod
    def calculate_area(p_image, combo):
        #   compute the area of the image that the patch takes up
        H, W = p_image.size()[-2:]
        _, area = MPA_agent.create_mask(
            combo,
            distributed_area=PA_cfg.dist_area,
            distributed_mask=PA_cfg.color,
            H=H, W=W,
        )
        return (area/(H*W)).item()

# ====================================================================
# HPA - Metropolis-Hastings Patch Attack (batch, probabilistic search)
# ====================================================================
class HPA():
    
    def __init__(self, dir_title):
        #   create a result directory path
        self.attack_dir = os.path.join(
            'PatchAttack_result',
            'HPA',
            dir_title,
            PA_cfg.HPA_attack_dir,
        )

    # --------------------------------------------------------------------------------
    #  Main method - run (or resume) the batch HPA attack over a specified index range
    # --------------------------------------------------------------------------------
    def attack(self, model, input_tensor, label_tensor, target, indices_range):
        
        #   create directories for each sample index
        attack_dirs = [os.path.join(self.attack_dir, '{}'.format(i)) for i in range(*indices_range)]
        
        #   see which indices have existing results and load them
        exist = []  #   boolean list for finished samples
        p_rcds = [] #   loaded records for finished sampled

        #   check each sample's directory for existence and records
        for attack_dir in attack_dirs:
            if not os.path.exists(attack_dir):
                os.makedirs(attack_dir) #   ensure directory exists

            if os.path.exists(os.path.join(attack_dir, 'finished.pt')):
                exist.append(True)
                loaded = torch.load(os.path.join(attack_dir, 'rcd.pt'), weights_only=False)
                p_rcds.append(edict(loaded))    #   store records
            else:
                exist.append(False)

        exist = torch.Tensor(exist).bool()  #   convert list to tensor
        load_num = sum(exist).item()        #   count existing records
        print('{} previous results loaded...'.format(load_num))
        #   if some samples are not processed, run the attack on them
        if load_num != len(exist):
            #   select the unfinished inputs
            masked_input_tensor = input_tensor[~exist]
            masked_label_tensor = label_tensor[~exist]
            #   prepare unfinished inputs on the GPU
            masked_target_tensor = torch.LongTensor(target).cuda(torch_cuda)[~exist]
            #   initialize the HPA agent
            hasting_attacker = HPA_agent(model)
            #   launch the HPA attack
            attacking_masks, acc, filters, area, queries, time_used = hasting_attacker(
                input_tensors=masked_input_tensor,
                label_tensors=masked_label_tensor,
                occlu_num=PA_cfg.n_occlu,
                sigma=PA_cfg.sigma,
                steps=PA_cfg.steps,
                t_atk=PA_cfg.target,
                target_tensors=masked_target_tensor if PA_cfg.target else None,
            )
            # the returned area is an average area over the batch
            # the returned queries is a summation of queries quired for the whole batch
            # the returned time_used is for the calculation of the whole batch

            #   Compute detailed occlusion areas for each sample
            _, areas_detail = HPA_agent.p.occlude_image(masked_input_tensor, attacking_masks)
            areas_detail = areas_detail / (input_tensor.size(-2)*input_tensor.size(-1))
            # areas_detail is a list storing all the areas for different inputs in the batch

            #   prepare lists of metrics for each unfinished sample
            t_masks = [*attacking_masks]
            t_areas = [*areas_detail]
            t_label_filter = [*filters[0]]
            t_target_filter = [*filters[1]] if filters[1] is not None else [None]*masked_input_tensor.size(0)
            t_queries = [queries/masked_input_tensor.size(0)]*masked_input_tensor.size(0)
            t_bs = [masked_input_tensor.size(0)]*masked_input_tensor.size(0)

            #   merge old and new records, saving the new ones
            old_i = 0
            new_i = 0
            rcds = []
            for i in range(len(attack_dirs)):
                if exist[i]:
                    rcds.append(p_rcds[old_i])
                    old_i+=1
                else:
                    tmp_rcd = edict()
                    tmp_rcd.mask = t_masks[new_i]
                    tmp_rcd.area = t_areas[new_i]
                    tmp_rcd.label_filter = t_label_filter[new_i]
                    tmp_rcd.target_filter = t_target_filter[new_i]
                    tmp_rcd.queries = t_queries[new_i]
                    tmp_rcd.bs = t_bs[new_i]
                    rcds.append(tmp_rcd)
                    new_i+=1
                    torch.save(dict(tmp_rcd), os.path.join(attack_dirs[i], 'rcd.pt'))
                    torch.save('finished!', os.path.join(attack_dirs[i], 'finished.pt'))
        else:
            #   all samples were already processed
            rcds = p_rcds

        #   compute batch level metrics
        avg_area = torch.mean(torch.tensor([item.area for item in rcds], dtype=torch.float32))
        non_target_success = 1 - torch.mean(torch.tensor([item.label_filter.item() for item in rcds], dtype=torch.float32))
        target_success = 1 - torch.mean(torch.tensor([item.target_filter.item() for item in rcds], dtype=torch.float32)) if PA_cfg.target else -1
        avg_queries = torch.mean(torch.tensor([item.queries for item in rcds], dtype=torch.float32))

        # print records
        print('*** Batch Size: {} | '
              'non_target_success rate: {:.4f} | target_success rate: {:.4f} | '
              'avg_queries: {:.4f} | avg_area_occluded: {:.4f} | '
              .format(
                  input_tensor.size(0),
                  non_target_success,
                  target_success,
                  avg_queries,
                  avg_area,
              ))

        # generate adv_image by applying recorded masks
        gen_masks = torch.stack([
            torch.from_numpy(item.mask) if isinstance(item.mask, np.ndarray) else item.mask
            for item in rcds
        ], dim=0)
        adv_image, areas_detail = HPA_agent.p.occlude_image(input_tensor, gen_masks)
        #   return the adverserial image and records
        return adv_image, rcds
        
    @staticmethod
    def calculate_areas(p_image, masks):
        #   compute the area of the image that's occluded for the given masks
        H, W = p_image.size()[-2:]
        _, areas = HPA_agent.p.occlude_image(p_image, masks)
        areas = areas / (H * W)
        return areas


class AP():

    def __init__(self, dir_title):
        
        dir_title = os.path.join(dir_title, PA_cfg.t_name)

        # attack dirs
        self.attack_dir = os.path.join(
            'PatchAttack_result',
            'AP',
            dir_title,
            PA_cfg.AP_attack_dir,
        )

    def attack(self, model, input_tensor, label_tensor, target, input_name='temp'):

        # load AdvPatch
        # check whether the AdvPatch has been built
        dir_to_load = os.path.join(
            PA_cfg.AdvPatch_dirs[target], 
            'patch_with_mask.pt',
        )
        if os.path.exists(dir_to_load):
            print('AdvPatch of label_{} is already built, loading...'.format(
            target))
            patch, mask = torch.load(dir_to_load, weights_only=True)
            patch, mask = patch.cuda(torch_cuda), mask.cuda(torch_cuda)
        else:
            assert False, 'AdvPatch of label {} not found, please generate it'.format(target)

        # set up attack-dirs
        attack_dir = os.path.join(self.attack_dir, input_name, 'patch_used_{}'.format(target))
        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)

        if os.path.exists(os.path.join(attack_dir, 'finished.pt')):
            loaded = torch.load(os.path.join(attack_dir, 'rcd.pt'), weights_only=True)
            rcd = edict(loaded)
            translate_tensor = rcd.translate_tensors[0]
            # generate adv_image
            temp_input = input_tensor
            temp_mask = mask.unsqueeze(0)
            temp_patch = patch.unsqueeze(0)
            mask_warpped = translate(temp_mask.float(), translate_tensor)
            patch_warpped = translate(temp_patch.float(), translate_tensor)
            overlay = temp_input * (1 - mask_warpped) + patch_warpped * mask_warpped
            print('Previous result loaded:')
        else:
            # set up records
            rcd = edict()
            rcd.translate_tensors = []
            rcd.masks = []
            rcd.areas = []
            rcd.target_success = []
            rcd.non_target_success = []

            # attack
            # the patch locates in the top left corner
            PATCH_SIZE = sum(mask[0,:,0]).item()
            H, W = input_tensor.size()[-2:]
            translate_space = [H-PATCH_SIZE+1, W-PATCH_SIZE+1]
            # random translation
            u_t = torch.randint(low=0, high=int(translate_space[0]), size=(1,)).item()
            v_t = torch.randint(low=0, high=int(translate_space[1]), size=(1,)).item()
            
            translate_tensor = torch.Tensor([u_t, v_t]).unsqueeze(0).cuda(torch_cuda)

            # translate patch
            temp_input = input_tensor
            temp_mask = mask.unsqueeze(0)
            temp_patch = patch.unsqueeze(0)

            mask_warpped = translate(temp_mask.float(), translate_tensor)
            patch_warpped = translate(temp_patch.float(), translate_tensor)
                
            # overlay
            overlay = temp_input * (1 - mask_warpped) + patch_warpped * mask_warpped

            # success check
            pred = model(overlay).argmax(dim=1)
            target_success = (pred==target).item()
            non_target_success = (pred!=label_tensor).item()

            # update rcd
            rcd.translate_tensors.append(translate_tensor)
            rcd.masks.append(mask_warpped)
            rcd.areas.append((mask_warpped.sum()/(H*W)).item())
            rcd.target_success.append(target_success)
            rcd.non_target_success.append(non_target_success)

            # save records
            torch.save(dict(rcd), os.path.join(attack_dir, 'rcd.pt'))

            # finished flag
            torch.save('finished!', os.path.join(attack_dir, 'finished.pt'))

        # print records
        print('*** '
              'non_target_success: {} | target_success: {} | occluded area: {:.4f} | '
              .format(
                  rcd.non_target_success,
                  rcd.target_success,
                  rcd.areas[0],
              )
        )
        return overlay, rcd

    @staticmethod
    def calculate_area(adv_image, mask):
        H, W = adv_image.size()[-2:]
        area = mask.sum() / (H*W)
        return area.item()

            




        


        