import numpy as np 
import math  
import cv2
import os    
import torch 
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
from network.modules import Modules as modules


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for PixelShufflePack."""
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
    


class ResidualBlocksWithInputConv2(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
    

class Temporal_Exposure_Correction_Module(nn.Module):
    def __init__(self, in_channels=3*2+1, out_channels=32, num_blocks=3):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        
        main.append(nn.Conv2d(out_channels, 3, 3, 1, 1, bias=True))

        self.main = nn.Sequential(*main)

    def forward(self, curr_lr, last_lr,ECG_feature):

        return self.main(torch.cat((curr_lr,last_lr,ECG_feature),1)) + curr_lr
    
    
    


class MMVSR(nn.Module):
    def __init__(self, mid_channels=32, num_blocks=30, spynet_pretrained=None):
        super().__init__()

        self.mid_channels = mid_channels

        self.TEC_module = Temporal_Exposure_Correction_Module()
        self.spynet = modules.SpyNet(os.path.abspath('./')+'/network-sintel-final.pytorch')

        self.compression_layer = ResidualBlocksWithInputConv(
            32 + 3, 32, num_blocks=2)
        

        self.edge_weightnet = nn.Sequential(
                        nn.Conv2d(mid_channels, 64, 1, 1, 0, bias=True),
                        ResidualBlockNoBN(64),
                        nn.Conv2d(64, 1, 1, 1, 0, bias=True)
                    )

        self.forward_resblocks = ResidualBlocksWithInputConv2(
            mid_channels, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 1, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self._raised_warning = False

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward
    

    def Dynamic_Static_Decoupled_Alignment(self,lr,dist_lr,feat_prop,dist_memory,device,Extreme_anomaly_flag,is_train):
        b, c, h, w = lr.size()
        Rlr_curr_rgb_avg = F.avg_pool2d(lr,5,1,2)
        Rlr_dist_rgb_avg = F.avg_pool2d(dist_lr,5,1,2)
        if is_train:
            patch_w = 6
            patch_h = 6
            batch=4
        else:
            patch_w = 32
            patch_h = 18
            batch=1

        forward_flow_3 = self.spynet(lr, dist_lr).view(b, 1, 2, h, w)
        Rlr_dist_rgb_warp = modules.flow_warp(dist_lr, forward_flow_3[:, 0, :, :, :].permute(0, 2, 3, 1))
        feat_prop_3_warp = modules.flow_warp(dist_memory, forward_flow_3[:, 0, :, :, :].permute(0, 2, 3, 1))
        
        flow_0_1_2_3 = forward_flow_3

        Rlr_dist_rgb_warp_avg = F.avg_pool2d(Rlr_dist_rgb_warp,5,1,2)
        Static_mask_1 = abs(Rlr_curr_rgb_avg-Rlr_dist_rgb_warp_avg)
        gating_threshold = torch.sum(Static_mask_1).cpu().item()/(b*c*h*w)
        if gating_threshold>0.5:
            return feat_prop.new_zeros(feat_prop.shape),Static_mask_1
        
        else:
            Static_mask_1 = torch.sum(Static_mask_1,1)
            Static_mask_1  = torch.where(Static_mask_1/3>0.03,0.,1.)
            Static_mask_1_array = Static_mask_1.cpu().numpy()
            for i in range(batch): 
                Static_mask_1_array_single = cv2.erode(Static_mask_1_array[i], np.ones((3,3),np.uint8), iterations = 2)
                if i ==0:
                    Static_mask_1 = torch.from_numpy(Static_mask_1_array_single).unsqueeze(0).unsqueeze(0)
                else: 
                    Static_mask_1 = torch.cat((Static_mask_1,torch.from_numpy(Static_mask_1_array_single).unsqueeze(0).unsqueeze(0)),0)
            Static_mask_1 = Static_mask_1.to(device) 

            feat_prop_3 = dist_memory.clone()
            for batch_index in range(batch): 
                dynamic_area = []
                for i in range(patch_w):
                    for j in range(patch_h):
                        img_res_avg_erode_patch = Static_mask_1[batch_index,:,j*10:j*10+10,i*10:i*10+10]
                        if torch.sum(img_res_avg_erode_patch).item()<95:
                            dynamic_area.append([i,j])

                for index in range(len(dynamic_area)):
                    i,j = dynamic_area[index]
                    Rlr_curr_rgb_avg_patch = Rlr_curr_rgb_avg[batch_index,:,j*10:j*10+10,i*10:i*10+10]
                    if math.isnan(torch.mean(flow_0_1_2_3[batch_index,:,0,j*10:j*10+10,i*10:i*10+10]).detach().item()):
                        flow_guide_x = 0
                    else:
                        flow_guide_x = round(torch.mean(flow_0_1_2_3[batch_index,:,0,j*10:j*10+10,i*10:i*10+10]).detach().item())
                    if math.isnan(torch.mean(flow_0_1_2_3[batch_index,:,1,j*10:j*10+10,i*10:i*10+10]).detach().item()):
                        flow_guide_y = 0
                    else:
                        flow_guide_y = round(torch.mean(flow_0_1_2_3[batch_index,:,1,j*10:j*10+10,i*10:i*10+10]).detach().item())
                    patch_list = []
                    match_loss_list = []
                    
                    for offset_x in range(flow_guide_x-3,flow_guide_x+3):
                        for offset_y in range(flow_guide_y-3,flow_guide_y+3):
                            Rlr_dist_rgb_avg_patch = Rlr_dist_rgb_avg[batch_index,:,j*10+offset_y:j*10+10+offset_y,i*10+offset_x:i*10+10+offset_x]
                            _,h,w = Rlr_dist_rgb_avg_patch.shape
                            if h<10 or w<10:
                                continue
                            
                            match_loss = torch.sum(abs(Rlr_dist_rgb_avg_patch-Rlr_curr_rgb_avg_patch)).detach().item()
                            match_loss_list.append(match_loss)
                            patch_list.append(dist_memory[batch_index,:,j*10+offset_y:j*10+10+offset_y,i*10+offset_x:i*10+10+offset_x])
                    if len(patch_list) !=0:
                        if min(match_loss_list)/100<0.1:
                            match_index = match_loss_list.index(min(match_loss_list))
                            feat_prop_3[batch_index,:,j*10:j*10+10,i*10:i*10+10] = patch_list[match_index]
                        else:
                            if Extreme_anomaly_flag==0:
                                feat_prop_3[batch_index,:,j*10:j*10+10,i*10:i*10+10] = feat_prop[batch_index,:,j*10:j*10+10,i*10:i*10+10]
                    else:
                        if Extreme_anomaly_flag==0:
                            feat_prop_3[batch_index,:,j*10:j*10+10,i*10:i*10+10] = feat_prop[batch_index,:,j*10:j*10+10,i*10:i*10+10]

            static_last_memory1 = feat_prop_3_warp*Static_mask_1 + feat_prop_3*(Static_mask_1-1)*-1
            return static_last_memory1, Static_mask_1



    def forward_start(self, lr):
        b, c, h, w = lr.size()

        feat_prop = lr.new_zeros(b, self.mid_channels, h, w)

        feat_prop = self.compression_layer(torch.cat((lr, feat_prop),1))
        multi_memory_list = [feat_prop, feat_prop,feat_prop]
        multi_memory = torch.cat([feat_prop, feat_prop,feat_prop], dim=1)

        b,c,h,w = multi_memory.shape
        multi_memory = multi_memory.view(b, 3, c//3, h, w)
        edge_weight_list = []
        for i in range(len(multi_memory_list)):
            edge_weight_list.append(self.edge_weightnet(multi_memory_list[i])) 

        edge_weight = torch.cat(edge_weight_list, dim=1) 

        edge_weight = edge_weight.view(b, 3, 1, h, w)
        edge_weight = F.softmax(edge_weight, dim=1) 
        multi_memory = torch.sum(multi_memory*edge_weight, dim=1, keepdim=False)

        feat_prop = self.forward_resblocks(multi_memory)

        out = self.lrelu(self.fusion(feat_prop))
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(lr)
        out += base

        return out,feat_prop
    
    def forward_converge(self, lr,lr_sequence,feat_prop,memory_sequence,flow_temporal_sequence,device,Extreme_anomaly_flag,memory_gate,is_train = True):
        b, c, h, w = lr.size()

        forward_flow = self.spynet(lr, lr_sequence[-2]).view(b, 1, 2, h, w)
        feat_prop = modules.flow_warp(feat_prop, forward_flow[:, 0, :, :, :].permute(0, 2, 3, 1))

        multi_memory_list = []
        memory_count = 0
        if memory_gate[0] ==1:
            feat_prop = self.compression_layer(torch.cat((lr, feat_prop),1))
            multi_memory_list.append(feat_prop)
            memory_count +=1
        if memory_gate[1] ==1:
            converged_memory_1, Static_mask_1 = self.Dynamic_Static_Decoupled_Alignment(lr,lr_sequence[-3],feat_prop,memory_sequence[-2],device,Extreme_anomaly_flag,is_train)
            converged_memory_1 = self.compression_layer(torch.cat((lr, converged_memory_1),1))
            multi_memory_list.append(converged_memory_1)
            memory_count +=1

        for i in range(3-memory_count):
            multi_memory_list = [multi_memory_list[0]]+multi_memory_list

        for i in range(len(multi_memory_list)):
            if i==0:
                multi_memory = multi_memory_list[i]
            else:
                multi_memory = torch.cat((multi_memory, multi_memory_list[i]),1)

        b,c,h,w = multi_memory.shape
        multi_memory = multi_memory.view(b, 3, c//3, h, w)
        edge_weight_list = []
        for i in range(len(multi_memory_list)):
            edge_weight_list.append(self.edge_weightnet(multi_memory_list[i]))

        edge_weight = torch.cat(edge_weight_list, dim=1)

        edge_weight = edge_weight.view(b, 3, 1, h, w)
        edge_weight = F.softmax(edge_weight, dim=1) 
        multi_memory = torch.sum(multi_memory*edge_weight, dim=1, keepdim=False)

        feat_prop = self.forward_resblocks(multi_memory)

        out = self.lrelu(self.fusion(feat_prop))
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(lr)
        out += base

        return out,feat_prop,forward_flow
    

    def forward_optical_flow(self, curr_lr, last_lr):
        b, c, h, w = curr_lr.size()
        forward_flow = self.spynet(curr_lr, last_lr).view(b, 1, 2, h, w)
        img_prop = modules.flow_warp(last_lr, forward_flow[:, 0, :, :, :].permute(0, 2, 3, 1))
        warp_error = torch.sum(abs(img_prop - curr_lr))/(h*w*3)
        return warp_error.cpu().item()


    def forward_EC(self, lr,lr_sequence,feat_prop,memory_sequence,flow_temporal_sequence,ECG_feature,device,is_train = True):
        b, c, h, w = lr.size()
        lr = self.TEC_module(lr, lr_sequence[-2],ECG_feature)
        forward_flow = self.spynet(lr, lr_sequence[-2]).view(b, 1, 2, h, w)
        feat_prop = modules.flow_warp(feat_prop, forward_flow[:, 0, :, :, :].permute(0, 2, 3, 1))
        feat_prop = self.compression_layer(torch.cat((lr, feat_prop),1))
        multi_memory_list = [feat_prop, feat_prop,feat_prop]
        multi_memory = torch.cat([feat_prop, feat_prop,feat_prop], dim=1)

        b,c,h,w = multi_memory.shape
        multi_memory = multi_memory.view(b, 3, c//3, h, w)
        edge_weight_list = []
        for i in range(len(multi_memory_list)):
            edge_weight_list.append(self.edge_weightnet(multi_memory_list[i]))

        edge_weight = torch.cat(edge_weight_list, dim=1)

        edge_weight = edge_weight.view(b, 3, 1, h, w)
        edge_weight = F.softmax(edge_weight, dim=1) 
        multi_memory = torch.sum(multi_memory*edge_weight, dim=1, keepdim=False)
        feat_prop = self.forward_resblocks(multi_memory)
        out = self.lrelu(self.fusion(feat_prop))
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(lr)
        out += base

        return out,feat_prop,forward_flow,lr
    


    def forward(self, lr,last_lr,feat_prop):
        b, c, h, w = lr.size()
        forward_flow = self.spynet(lr, last_lr).view(b, 1, 2, h, w)
        feat_prop = modules.flow_warp(feat_prop, forward_flow[:, 0, :, :, :].permute(0, 2, 3, 1))
        feat_prop = self.compression_layer(torch.cat((lr, feat_prop),1))
        multi_memory_list = [feat_prop, feat_prop,feat_prop]
        multi_memory = torch.cat([feat_prop, feat_prop,feat_prop], dim=1)

        b,c,h,w = multi_memory.shape
        multi_memory = multi_memory.view(b, 3, c//3, h, w)
        edge_weight_list = []
        for i in range(len(multi_memory_list)):
            edge_weight_list.append(self.edge_weightnet(multi_memory_list[i]))

        edge_weight = torch.cat(edge_weight_list, dim=1)

        edge_weight = edge_weight.view(b, 3, 1, h, w)
        edge_weight = F.softmax(edge_weight, dim=1) 
        multi_memory = torch.sum(multi_memory*edge_weight, dim=1, keepdim=False)

        feat_prop = self.forward_resblocks(multi_memory)
        out = self.lrelu(self.fusion(feat_prop))
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(lr)
        out += base

        return out,feat_prop,forward_flow
    








        
        









