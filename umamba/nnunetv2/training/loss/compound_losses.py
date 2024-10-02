import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from nnunetv2.training.loss.cbdice_loss import SoftcbDiceLoss, SoftclMDiceLoss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

# Mapping dictionary for regions # this is for AortaSeg2024 challenge
label_mapping = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2,
    10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4
}

# Function to remap labels
def remap_labels(tensor):
    """
    Remaps labels in the given tensor according to the label_mapping dictionary.
    
    Args:
        tensor (torch.Tensor): The input tensor with shape (b, 1, x, y, z) for target 
                               or (b, 24, x, y, z) for predictions.
                               
    Returns:
        torch.Tensor: The tensor with remapped labels.
    """
    b, c, x, y, z = tensor.shape

    if c == 1:  # Target tensor
        remapped_tensor = tensor.clone()
        for old_label, new_label in label_mapping.items():
            remapped_tensor[tensor == old_label] = new_label
    else:  # Prediction tensor
        region_tensor = torch.zeros(b, 5, x, y, z, device=tensor.device, dtype=tensor.dtype)  # Assuming 5 regions (0 background + 4 regions)
        for old_label, new_label in label_mapping.items():
            region_tensor[:, new_label] += tensor[:, old_label]

        remapped_tensor = region_tensor

    return remapped_tensor
class Region_DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_region:
        :param weight_target:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight_region: float):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Remap predictions and target to regions
        region_net_output = remap_labels(net_output)
        region_target = remap_labels(target)
        region_target_dice = region_target
        
        # Ensure region_net_output is of float type for softmax
        #region_net_output = region_net_output.to(torch.float32)
        
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        region_dc_loss = self.dc(region_net_output, region_target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        region_ce_loss = self.ce(region_net_output, region_target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        target_result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        region_result = self.weight_ce * region_ce_loss + self.weight_dice * region_dc_loss
        result = weight_region * region_result + (1-weight_region) * target_result
        return result
    
class Region_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1.2, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Region_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Remap predictions and target to regions
        region_net_output = remap_labels(net_output)
        region_target = remap_labels(target)
        region_target_dice = region_target
        
        # Ensure region_net_output is of float type for softmax
        #region_net_output = region_net_output.to(torch.float32)
        
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        region_dc_loss = self.dc(region_net_output, region_target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        region_ce_loss = self.ce(region_net_output, region_target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        target_result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        region_result = self.weight_ce * region_ce_loss + self.weight_dice * region_dc_loss
        #result = ( region_result+  target_result)/2
        result = (region_result*0.2) +  (target_result*0.8)
        return result
    
class Region_DC_and_CE_and_CBDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cbdc_kwargs, weight_ce=1, weight_dice=1, weight_cbdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Region_DC_and_CE_and_CBDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cbdice = weight_cbdice
        
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cbdice = SoftcbDiceLoss(**cbdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Remap predictions and target to regions
        region_net_output = remap_labels(net_output)
        region_target = remap_labels(target)
        region_target_dice = region_target
        
        # Ensure region_net_output is of float type for softmax
        #region_net_output = region_net_output.to(torch.float32)
        
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        region_dc_loss = self.dc(region_net_output, region_target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        region_ce_loss = self.ce(region_net_output, region_target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        #target_result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        region_result = self.weight_ce * region_ce_loss + self.weight_dice * region_dc_loss
        cbdice_loss = self.cbdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cbdice != 0 else 0
        target_result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cbdice * cbdice_loss
        
        result = (region_result*0.2) +  (target_result*0.8)
        return result
    
    
class DC_and_CE_and_CBDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cbdc_kwargs, weight_ce=1, weight_dice=1, weight_cbdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Original from: https://github.com/PengchengShi1220/cbDice/tree/main
        
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_CBDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cbdice = weight_cbdice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cbdice = SoftcbDiceLoss(**cbdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        cbdice_loss = self.cbdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cbdice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cbdice * cbdice_loss
        return result
    