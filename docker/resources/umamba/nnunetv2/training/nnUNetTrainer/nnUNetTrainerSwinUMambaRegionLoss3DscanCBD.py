import numpy as np
from time import time
from os.path import join

import torch
from torch import distributed as dist
from torch import autocast, nn

from typing import Union, Tuple, List

from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger, nnUNetLogger_Aorta2024
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
#from nnunetv2.nets.SwinUmamba3D import get_swin_umamba_from_plans
from nnunetv2.nets.SwinUmamba3DSS3Dsmall import get_swin_umamba_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.loss.compound_losses import Region_DC_and_CE_and_CBDC_loss, Region_DC_and_CE_loss,DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import  MemoryEfficientSoftDiceLoss


# This is specifically for AortaSeg2024 challenge.
label_mapping = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2,
    10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4
}

class nnUNetTrainerSwinUMambaRegionLoss3DscanCBD(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        #self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        #self.freeze_encoder_epochs = 10
       #self.early_stop_epoch = 350
        self.logger = nnUNetLogger_Aorta2024()
        self.num_epochs=1600
        self.new_num_epochs=self.num_epochs
        
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_swin_umamba_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 3D models are supported")

        
        print("SwinUMamba: {}".format(model))

        return model
    def train_step(self, batch: dict) -> dict:

            data = batch['data']
            target = batch['target']

            data = data.to(self.device, non_blocking=True)
            if isinstance(target, list):
                target = [i.to(self.device, non_blocking=True) for i in target]
            else:
                target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            # Autocast is a little bitch.
            # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)
                # del data
                l = self.loss(output, target)

            if self.grad_scaler is not None:
                self.grad_scaler.scale(l).backward()
                self.grad_scaler.unscale_(self.optimizer)
                #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                grad_norm  = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                l.backward()
                #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                grad_norm  = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            if grad_norm > 100:
                self.print_to_log_file(f"->>> extreme Large gradient norm: {grad_norm}")
                

            return {'loss': l.detach().cpu().numpy()}
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss =  Region_DC_and_CE_and_CBDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'iter_': 10, 'smooth': 1e-3}, weight_ce=1, weight_dice=1, weight_cbdice=1, ignore_label=self.label_manager.ignore_label, 
                                    dice_class=MemoryEfficientSoftDiceLoss)

        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss
    
    def remap_outputs_to_regions(self,tp, fp, fn):
        """
        Remaps the outputs to regions based on the label mapping.
        """
        #print(len(tp))
        region_tp = np.zeros(4, dtype=np.int64)  # Assuming 5 regions (4 regions)
        region_fp = np.zeros(4, dtype=np.int64)
        region_fn = np.zeros(4, dtype=np.int64)
        #print("label_mapping",label_mapping)
        #print("label_mapping.items()",label_mapping.items())
        for old_label, new_label in label_mapping.items():
            #print("old_label->new_label",old_label, new_label)
            region_tp[new_label-1] += tp[old_label-1]
            region_fp[new_label-1] += fp[old_label-1]
            region_fn[new_label-1] += fn[old_label-1]
            
        return region_tp, region_fp, region_fn


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        
        # Remap to regions and calculate Dice score for regions
        region_tp, region_fp, region_fn = self.remap_outputs_to_regions(tp, fp, fn)
        global_dc_per_region = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else np.nan for i, j, k in zip(region_tp, region_fp, region_fn)]
        mean_region_dice = np.nanmean(global_dc_per_region)
    
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('mean_region_dice', mean_region_dice, self.current_epoch)
        self.logger.log('dice_per_region', global_dc_per_region, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)


    def on_epoch_end(self):
        region_dice = [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_region'][-1]]
        mean_region_dice = np.round(self.logger.my_fantastic_logging['mean_region_dice'][-1], decimals=4)

        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Mean region dice', mean_region_dice)
        self.print_to_log_file('Region dice', region_dice)
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.new_num_epochs- 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        
    # def on_train_epoch_start(self):
    #     # freeze the encoder if the epoch is less than 10
    #     if self.current_epoch < self.freeze_encoder_epochs:
    #         self.print_to_log_file("Freezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.freeze_encoder()
    #         else:
    #             self.network.freeze_encoder()
    #     else:
    #         self.print_to_log_file("Unfreezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.unfreeze_encoder()
    #         else:
    #             self.network.unfreeze_encoder()
    #     super().on_train_epoch_start()
        
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0,1.0,1.0], [0.5,0.5,0.5], [0.25,0.25,0.25], [0.125,0.125,0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales