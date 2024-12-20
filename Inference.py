import os
import argparse
import random
import numpy as np
import torchvision 
import torch
from torch.utils.data import DataLoader 
from torchvision import utils as vutils
import torch.nn.functional as F
from network import MMVSR
from Our_dataloader import val_GetData
from evaluation_metrics import calc_psnr,calc_ssim,calc_lpips,lpips_fun

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers.')
    parser.add_argument('--pretrain', action='store_true', default=True, help='Use pretrain model.')
    parser.add_argument('--pretrian_model_path', type=str, default='/home/amax-2/TGZ/Video_SR/MMVSR/save_model/Pretrain_MMVSR.pkl', help='pretrain model path.')
    parser.add_argument('--experiment_index', type=str, default='default', help='the experiment_index.')
    parser.add_argument('--rondom_seed', type=int, default=1, help='rondom seed.') 
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

def evolution(cfg,Data_path,net,valloader_list,DEVICE):
    net.eval()
    valloader_name_list = ['000','011','015','020']
    PSNR_all = [] 
    SSIM_all = [] 
    LPIPS_all = []
    
    for i in range(len(valloader_list)):

        PSNR_single = [] 
        SSIM_single = [] 
        LPIPS_single = [] 
        lightness_normal_list = []
        lightness_unnormal_list = []
        memory_temporal_sequence = []
        flow_temporal_sequence = []
        lr_temporal_sequence = []

        for iter, (hr, lr) in enumerate(valloader_list[i]): 
            hr = hr.to(DEVICE).float()
            lr = lr.to(DEVICE).float()

            lightness_normal_list.append(torch.mean(lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).item())
            lightness_unnormal_list.append(torch.mean(lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).item())

            if iter>0 and iter % 10 == 0:
                if iter % 20 == 0:
                    gamma = 0.25
                    lr_array = lr.cpu().numpy()
                    lr_array = np.clip(np.power(lr_array,gamma),0.,1.)
                    lr = torch.from_numpy(lr_array).to(DEVICE).float()
                else:
                    gamma = 2.3
                    lr_array = lr.cpu().numpy()
                    lr_array = np.clip(np.power(lr_array,gamma),0.,1.)
                    lr = torch.from_numpy(lr_array).to(DEVICE).float()
                lightness_unnormal_list = lightness_unnormal_list[:-1]
                lightness_unnormal_list.append(torch.mean(lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).item())

            else:
                lr = lr.to(DEVICE).float()

            lr_temporal_sequence.append(lr)
            with torch.no_grad():
                if iter==0:
                    curr_lr_Y = (lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).unsqueeze(1)
                    generate_SR,memory_bank = net.forward_start(lr)
                    memory_temporal_sequence.append(memory_bank)
                else:
                    optical_flow_error = net.forward_optical_flow(lr,lr_temporal_sequence[-2])

                    if iter>=10 and abs(optical_flow_error)>0.08:
                        curr_lr_Y = (lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).unsqueeze(1) # [n,1,h,w]
                        last_lr_Y = (lr_temporal_sequence[-2][:,0,:,:]*0.299+lr_temporal_sequence[-2][:,1,:,:]*0.587+lr_temporal_sequence[-2][:,2,:,:]*0.114).unsqueeze(1) # [n,1,h,w]
                        curr_Y_avg_pool = F.avg_pool2d(curr_lr_Y,11,1,5)
                        curr_Y_avg_pool = F.avg_pool2d(curr_Y_avg_pool,11,1,5)
                        last_Y_avg_pool = F.avg_pool2d(last_lr_Y,11,1,5)
                        last_Y_avg_pool = F.avg_pool2d(last_Y_avg_pool,11,1,5)
                        ECG_feature = curr_Y_avg_pool-last_Y_avg_pool 

                        generate_SR,memory_bank,optical_flow,Ec_lr = net.forward_EC(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,ECG_feature,DEVICE,is_train=False)
                        memory_temporal_sequence.append(memory_bank)
                        flow_temporal_sequence.append(optical_flow)
                        lr_temporal_sequence = lr_temporal_sequence[:-1]
                        lr_temporal_sequence.append(Ec_lr)
                        vutils.save_image(lr, './save_model/'+cfg.experiment_index+'/AE_lr'+'_'+valloader_name_list[i]+'_'+str(iter)+'_'+cfg.experiment_index+'.png')
                        vutils.save_image(Ec_lr, './save_model/'+cfg.experiment_index+'/Ec_lr'+'_'+valloader_name_list[i]+'_'+str(iter)+'_'+cfg.experiment_index+'.png')


                    elif iter>=10 and iter%9==0:
                        memory_gate=[1,1,1]
                        curr_lr_Y = lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114 # [n,h,w]
                        generate_SR,memory_bank,optical_flow = net.forward_converge(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,DEVICE,0,memory_gate,is_train=False)
                        memory_temporal_sequence.append(memory_bank)
                        flow_temporal_sequence.append(optical_flow)  

                    else:
                        generate_SR,memory_bank,optical_flow = net.forward(lr,lr_temporal_sequence[-2],memory_bank)
                        memory_temporal_sequence.append(memory_bank)
                        flow_temporal_sequence.append(optical_flow)   
                    
            vutils.save_image(generate_SR, './save_model/'+cfg.experiment_index+'/SR'+'_'+valloader_name_list[i]+'_'+str(iter)+'_'+cfg.experiment_index+'.png')
            
            hr_array = np.transpose(hr.cpu().numpy()[0], (1,2,0))
            generate_SR_array = np.transpose(generate_SR.cpu().numpy()[0], (1,2,0))
            PSNR_our = calc_psnr(hr_array*255, generate_SR_array*255)
            SSIM_our = calc_ssim(hr_array*255, generate_SR_array*255) 
            lpips_our = calc_lpips(hr,generate_SR,lpips_fun)

            PSNR_single.append(PSNR_our)
            SSIM_single.append(SSIM_our)
            LPIPS_single.append(lpips_our)
            
            print(str(iter+1)+' / '+str(100))
        PSNR_value, SSIM_value, LPIPS_value = sum(PSNR_single)/len(PSNR_single), sum(SSIM_single)/len(SSIM_single), sum(LPIPS_single)/len(LPIPS_single)
        print(Data_path+'_'+valloader_name_list[i]+'_PSNR: ',PSNR_value)
        print(Data_path+'_'+valloader_name_list[i]+'_SSIM: ',SSIM_value)
        print(Data_path+'_'+valloader_name_list[i]+'_LPIPS: ',LPIPS_value) 
        with open('./log/log_PSNR_SSIM_LPIPS_'+cfg.experiment_index+'.txt',"a") as f2:
            f2.write('Clip: '+valloader_name_list[i]+' | '+Data_path+' | '+'PSNR: '+str(PSNR_value)+' | '+'SSIM: '+str(SSIM_value)+' | '+'LPIPS: '+str(LPIPS_value)+'\n')
        PSNR_all.append(PSNR_value)
        SSIM_all.append(SSIM_value)
        LPIPS_all.append(LPIPS_value)

    PSNR_all_value, SSIM_all_value, LPIPS_all_value = sum(PSNR_all)/len(PSNR_all), sum(SSIM_all)/len(SSIM_all), sum(LPIPS_all)/len(LPIPS_all)
    with open('./log/log_PSNR_SSIM_LPIPS_'+cfg.experiment_index+'.txt',"a") as f2:
            f2.write('REDS4: '+' | '+Data_path+' | '+'PSNR: '+str(PSNR_all_value)+' | '+'SSIM: '+str(SSIM_all_value)+' | '+'LPIPS: '+str(LPIPS_all_value)+'\n')
            f2.write('\n')
    return PSNR_all_value, SSIM_all_value, LPIPS_all_value

def main(cfg,DEVICE):
    if cfg.pretrain:
        net = MMVSR.MMVSR().to(DEVICE)
        net.load_state_dict(torch.load(cfg.pretrian_model_path))
        net.eval() 
        print('Load pretrain model complete.') 

    val_file_name_list = sorted(os.listdir('./dataset/REDS/test_sharp'))
    valset_000 = val_GetData('./dataset/REDS',val_file_name_list[0])
    valloader_000 = DataLoader(valset_000, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    valset_011 = val_GetData('./dataset/REDS',val_file_name_list[1])
    valloader_011 = DataLoader(valset_011, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    valset_015 = val_GetData('./dataset/REDS',val_file_name_list[2])
    valloader_015 = DataLoader(valset_015, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    valset_020 = val_GetData('./dataset/REDS',val_file_name_list[3])
    valloader_020 = DataLoader(valset_020, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    evolution(cfg,'REDS4',net,[valloader_000,valloader_011,valloader_015,valloader_020],DEVICE)
    torch.cuda.empty_cache() 
                    

if __name__ == "__main__":
    cfg = parse_args()
    setup_seed(cfg.rondom_seed)
    print(torch.__version__)
    print(torchvision.__version__)
    print (os.getcwd()) 
    print (os.path.abspath('..'))
    
    if not os.path.exists(os.path.join('./save_model',cfg.experiment_index)):
        os.makedirs(os.path.join('./save_model',cfg.experiment_index))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cfg,DEVICE)
