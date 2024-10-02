
from torch import nn
import torch
from os.path import join
from typing import Union, Tuple, List
import numpy as np 
from time import time
from torch import distributed as dist

from nnunetv2.utilities.collate_outputs import collate_outputs

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger, nnUNetLogger_Aorta2024_Distill
from nnunetv2.nets.UMambaEnc_3dLarge import get_umamba_enc_3d_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler25 import ConfigurationManager, PlansManager
from nnunetv2.utilities.label_handling.label_handling25 import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss, Region_DC_and_CE_loss,DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import  MemoryEfficientSoftDiceLoss
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.run import load_pretrained_weights
from torch import autocast, nn

# This is specifically for AortaSeg2024 challenge.
label_mapping = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2,
    10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4
}

class nnUNetTrainerUMambaEncRegionLossDistill(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        #self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        #self.freeze_encoder_epochs = 10
       #self.early_stop_epoch = 350
        self.logger = nnUNetLogger_Aorta2024_Distill() #nnUNetLogger_Aorta2024()
        self.num_epochs=1200
        self.new_num_epochs=self.num_epochs
        self.teacher = None
    
    def init_teacher(self):
        predictor = nnUNetPredictor(
                tile_step_size=0.6,
                use_gaussian=False,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=torch.device('cuda', 0),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
                ds=True
            )
        nnUNet_results = '/processing/jintao/nnUNet_results/Dataset260_AortaSeg24/'
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'nnUNetTrainerUMambaEncRegionLoss__nnUNetPlans__3d_fullres'),
            use_folds=('all',),
            checkpoint_name='checkpoint_best.pth'
        )
        
        self.teacher = predictor.network
        
        #checkpoint_fn = '/processing/jintao/nnUNet_results/Dataset260_AortaSeg24/nnUNetTrainerUMambaEncRegionLoss__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth'
        #self.teacher = load_pretrained_weights(self.teacher,checkpoint_fn)
        
        # Force disable gradients for the teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None
        self.teacher.eval()
        
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            if self.teacher is None:
                self.init_teacher()
            self.teacher.eval()
            self.teacher.to(self.device)
            
            if self.check_grad_disabled(self.teacher):
                self.print_to_log_file("WARNING: Some parameters in the teacher model still require gradients!")
                # Optionally, you can forcibly disable gradients here:
                for param in self.teacher.parameters():
                    param.requires_grad = False
            else:
                self.print_to_log_file("All teacher model parameters have requires_grad=False")
        
        
            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                self.enable_deep_supervision,
            ).to(self.device)
            
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
            

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        # loss = Region_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
        #                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
        #                           ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
            
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
    
    def distillation_loss(self, student_outputs, teacher_outputs, temperature=1.0):
        def softmax_with_temperature(logits, temperature):
            return torch.nn.functional.softmax(logits / temperature, dim=1)

        def log_softmax_with_temperature(logits, temperature):
            return torch.nn.functional.log_softmax(logits / temperature, dim=1)

        distillation_losses = []
        for student_out, teacher_out in zip(student_outputs, teacher_outputs):
            # Debug: Print raw logits
            # print(f"Student Logits: {student_out}")
            # print(f"Teacher Logits: {teacher_out}")

            student_log_prob = log_softmax_with_temperature(student_out, temperature)
            teacher_prob = softmax_with_temperature(teacher_out, temperature)

            # Remove clamping for now to avoid potential issues
            # epsilon = 1e-6
            # student_log_prob = torch.clamp(student_log_prob, min=epsilon, max=1.0 - epsilon)
            # teacher_prob = torch.clamp(teacher_prob, min=epsilon, max=1.0 - epsilon)

            # Debugging: Log intermediate values
            # print(f"Student Log Prob: {student_log_prob.mean().item()}")
            # print(f"Teacher Prob: {teacher_prob.mean().item()}")

            # KL Divergence between teacher and student
            kl_div = torch.nn.functional.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
            #print(f"KL Divergence: {kl_div.item()}")
            distillation_losses.append(kl_div * (temperature ** 2))
        
        # Average loss over all outputs (e.g., for deep supervision)
        return sum(distillation_losses) / len(distillation_losses)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        print(configuration_manager)
        if len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model
    
    def get_current_alpha(self, epoch):
        if epoch < 100:
            return 0.1
        elif epoch < 500:
            return 0.5        
        elif epoch < 800:
            return 0.3
        else:
            return 0.2
        
    def check_grad_disabled(self,model):
        for param in model.parameters():
            if param.requires_grad:
                print("Found parameter with requires_grad=True in teacher model")
                return False
        return True    

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            teacher_output = [output.detach() for output in self.teacher(data)]  # Detach each tensor in the list

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            student_output = self.network(data)
            # Calculate original loss
            original_loss = self.loss(student_output, target)
            # Calculate distillation loss
            distill_loss = self.distillation_loss(student_output, teacher_output) #self.loss(teacher_output, target)
            
        # Combine losses
        alpha =self.get_current_alpha(self.current_epoch)
        
        total_loss = (1 - alpha) * original_loss + alpha * distill_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
            
        assert torch.isfinite(data).all(), "Input data contains NaN or Inf values."
        for i, output in enumerate(teacher_output):
            assert torch.isfinite(output).all(), f"Teacher network output at index {i} contains NaN or Inf values."
        for i, output in enumerate(teacher_output):
            assert torch.isfinite(output).all(), f"Student network output at index {i} contains NaN or Inf values."


        return {
            'loss': total_loss.detach().cpu().numpy(),
            'original_loss': original_loss.detach().cpu().numpy(),
            'distill_loss': distill_loss.detach().cpu().numpy()
        }
        
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()

            original_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(original_losses_tr, outputs['original_loss'])
            original_loss_here = np.vstack(original_losses_tr).mean()

            distill_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(distill_losses_tr, outputs['distill_loss'])
            distill_loss_here = np.vstack(distill_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            original_loss_here = np.mean(outputs['original_loss'])
            distill_loss_here = np.mean(outputs['distill_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_original_losses', original_loss_here, self.current_epoch)
        self.logger.log('train_distill_losses', distill_loss_here, self.current_epoch)
    
        
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
        
    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        region_dice = [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_region'][-1]]
        mean_region_dice = np.round(self.logger.my_fantastic_logging['mean_region_dice'][-1], decimals=4)

        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)


        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('train_original_loss', np.round(self.logger.my_fantastic_logging['train_original_losses'][-1], decimals=4))
        self.print_to_log_file('train_distill_loss', np.round(self.logger.my_fantastic_logging['train_distill_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Mean region dice', mean_region_dice)
        self.print_to_log_file('Region dice', region_dice)
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.new_num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        
        
    def run_training(self):
        self.on_train_start()
        
        if self.check_grad_disabled(self.teacher):
            self.print_to_log_file("WARNING: Teacher model parameters require gradients on train start start!")
            for param in self.teacher.parameters():
                param.requires_grad = False

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
