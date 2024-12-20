import os
import argparse
import random
import numpy as np
import time
import torchvision 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision import utils as vutils
import torch.nn.functional as F
from torch.optim import lr_scheduler
from network import MMVSR
from Our_dataloader import train_GetData,val_GetData
from evaluation_metrics import calc_psnr,calc_ssim,calc_lpips,lpips_fun

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", type=int, default=1)
    parser.add_argument("--Epoch", type=int, default=4600)
    parser.add_argument("--REDS_h", type=int, default=180)
    parser.add_argument("--REDS_w", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers.')
    parser.add_argument("--LR", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--use_L2', action='store_true', default=False, help='Use L2 loss instead of L1 loss.')
    parser.add_argument('--Lambda_1', type=float, default=1., help='L1 loss weight.')
    parser.add_argument('--Lambda_2', type=float, default=0.5, help='CE loss weight.')
    parser.add_argument('--pretrain', action='store_true', default=False, help='Use pretrain model.')
    parser.add_argument('--pretrain_model_path', type=str, default='./', help='pretrain model path.')
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

def evolution(cfg,Data_path,net,epoch,valloader_list,DEVICE):
    net.eval()
    valloader_name_list = ['000','011','015','020']
    PSNR_all = [] 
    SSIM_all = [] 
    LPIPS_all = []
    for i in range(len(valloader_list)):
        PSNR_single = [] 
        SSIM_single = [] 
        LPIPS_single = [] 
        memory_temporal_sequence = []
        flow_temporal_sequence = []
        lr_temporal_sequence = []
        for iter, (hr, lr) in enumerate(valloader_list[i]): #,Lr_input_hf
            hr = hr.to(DEVICE).float()
            lr = lr.to(DEVICE).float()
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

            else:
                lr = lr.to(DEVICE).float()
            lr_temporal_sequence.append(lr)
            with torch.no_grad():
                if iter==0:
                    curr_lr_Y = (lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).unsqueeze(1)
                    generate_SR,memory_bank = net.forward_start(lr)
                    memory_temporal_sequence.append(memory_bank)
                else:

                    if iter>0 and iter % 10 == 0:
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

                    elif iter%3==0:
                        memory_gate=[1,1,1]
                        curr_lr_Y = lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114
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
 
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def main(cfg,DEVICE):
    net = MMVSR.MMVSR().to(DEVICE)
    if cfg.pretrain:
        net.load_state_dict = torch.load(cfg.pretrain_model_path)
        net.train()
        print('Load pretrain teacher model complete.') 

    if cfg.train_flag == 1:
        with open('./log/log_training_setting'+cfg.experiment_index+'.txt',"a") as f:
            f.write('Total epochs: '+str(cfg.Epoch)+'\n')
            f.write('Batch size: '+str(cfg.batch_size)+'\n')
            f.write('Crop size: '+str(cfg.crop_size)+'\n')
            f.write('Initial learning rate: '+str(cfg.LR)+'\n')
            f.write('L1/2 loss weight: '+str(cfg.Lambda_1)+'\n')
            f.write('Use pretrain model: '+str(cfg.pretrain)+'\n')
            f.write('random seed: '+str(cfg.rondom_seed)+'\n')

        begin_time = time.time()
        net.train()
        spynet_id = list(map(id, net.spynet.parameters()))
        base_params = filter(lambda p: id(p) not in spynet_id, net.parameters())
        optimizer_G = torch.optim.Adam([
            {'params': base_params},
            {'params': net.spynet.parameters(),'lr':2.5e-5}],
            lr=cfg.LR,
            betas=(0.9, 0.99)
            )
        
        scheduler_G = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G,T_0=300000,eta_min=1e-7)

        if cfg.use_L2:
            criterion_pixelwise = nn.MSELoss()
        else:
            criterion_pixelwise = L1_Charbonnier_loss()

        train_file_name_list = sorted(os.listdir('./dataset/REDS/train_sharp'))
        train_len = len(train_file_name_list)

        val_file_name_list = sorted(os.listdir('./dataset/REDS/test_sharp'))
        valset_000 = val_GetData('./dataset/REDS',val_file_name_list[0])
        valloader_000 = DataLoader(valset_000, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        valset_011 = val_GetData('./dataset/REDS',val_file_name_list[1])
        valloader_011 = DataLoader(valset_011, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        valset_015 = val_GetData('./dataset/REDS',val_file_name_list[2])
        valloader_015 = DataLoader(valset_015, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        valset_020 = val_GetData('./dataset/REDS',val_file_name_list[3])
        valloader_020 = DataLoader(valset_020, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


        count_iter=0
        for epoch_count in range(cfg.Epoch):
            random.shuffle(train_file_name_list) 
            for clip_index in range(train_len//4):
                crop_x = random.randint(0, cfg.REDS_w-cfg.crop_size)
                crop_y = random.randint(0, cfg.REDS_h-cfg.crop_size)
                flipud_flag = random.random()
                fliplr_flag = random.random()
                angle_flag = random.choice([0, 1, 2, 3])

                trainset = train_GetData('./dataset/REDS',train_file_name_list[clip_index*4],train_file_name_list[clip_index*4+1],train_file_name_list[clip_index*4+2],train_file_name_list[clip_index*4+3],
                                                          cfg.crop_size,crop_x,crop_y,flipud_flag,fliplr_flag,angle_flag)
                
                trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)

                hr_temporal_sequence = []
                lr_temporal_sequence = []
                memory_temporal_sequence = []
                flow_temporal_sequence = []
                loss_pixel = 0
                loss_EC = torch.from_numpy(np.zeros(1)).to(DEVICE)
                AE_flag = 0 
                OUE_flag = random.randint(0,1) 
                if count_iter <= 1000:
                    Extreme_anomaly_flag=0
                else:
                    if Extreme_anomaly_flag == 0:
                        Extreme_anomaly_flag = 1
                    else:
                        Extreme_anomaly_flag = 0

                count_iter += 1
                pixel_loss_count = 0
                ec_loss_count = 0

                for iter, (hr1,lr1,hr2,lr2,hr3,lr3,hr4,lr4) in enumerate(trainloader):


                    hr1 = hr1.to(DEVICE).float()
                    lr1 = lr1.to(DEVICE).float()
                    hr2 = hr2.to(DEVICE).float()
                    lr2 = lr2.to(DEVICE).float()
                    hr3 = hr3.to(DEVICE).float()
                    lr3 = lr3.to(DEVICE).float()
                    hr4 = hr4.to(DEVICE).float()
                    lr4 = lr4.to(DEVICE).float()
                
                    if iter==7:
                        if OUE_flag: 
                            gamma = round(random.uniform(0.25,0.5),2)
                            lr_array1 = lr1.cpu().numpy()
                            lr_array1 = np.clip(np.power(lr_array1,gamma),0.,1.)
                            lr1_EA = torch.from_numpy(lr_array1).to(DEVICE).float()

                            lr_array2 = lr2.cpu().numpy()
                            lr_array2 = np.clip(np.power(lr_array2,gamma),0.,1.)
                            lr2_EA = torch.from_numpy(lr_array2).to(DEVICE).float()

                            lr_array3 = lr3.cpu().numpy()
                            lr_array3 = np.clip(np.power(lr_array3,gamma),0.,1.)
                            lr3_EA = torch.from_numpy(lr_array3).to(DEVICE).float()

                            lr_array4 = lr4.cpu().numpy()
                            lr_array4 = np.clip(np.power(lr_array4,gamma),0.,1.)
                            lr4_EA = torch.from_numpy(lr_array4).to(DEVICE).float()
                        else: 
                            gamma = 3-round(random.uniform(0.,1),2)
                            lr_array1 = lr1.cpu().numpy()
                            lr_array1 = np.clip(np.power(lr_array1,gamma),0.,1.)
                            lr1_EA = torch.from_numpy(lr_array1).to(DEVICE).float()

                            lr_array2 = lr2.cpu().numpy()
                            lr_array2 = np.clip(np.power(lr_array2,gamma),0.,1.)
                            lr2_EA = torch.from_numpy(lr_array2).to(DEVICE).float()

                            lr_array3 = lr3.cpu().numpy()
                            lr_array3 = np.clip(np.power(lr_array3,gamma),0.,1.)
                            lr3_EA = torch.from_numpy(lr_array3).to(DEVICE).float()

                            lr_array4 = lr4.cpu().numpy()
                            lr_array4 = np.clip(np.power(lr_array4,gamma),0.,1.)
                            lr4_EA = torch.from_numpy(lr_array4).to(DEVICE).float()

                        hr = torch.cat((hr1,hr2,hr3,hr4),0)
                        lr = torch.cat((lr1_EA,lr2_EA,lr3_EA,lr4_EA),0)
                        lr_original = torch.cat((lr1,lr2,lr3,lr4),0)

                    else:
                        hr = torch.cat((hr1,hr2,hr3,hr4),0) 
                        lr = torch.cat((lr1,lr2,lr3,lr4),0) 


                    hr_temporal_sequence.append(hr)
                    lr_temporal_sequence.append(lr)

                    if iter==0:
                        sr,memory_bank = net.forward_start(lr)
                        loss_pixel += criterion_pixelwise(hr, sr)
                        pixel_loss_count += 1
                        memory_temporal_sequence.append(memory_bank)

                    else:
                        curr_lr_Y = (lr[:,0,:,:]*0.299+lr[:,1,:,:]*0.587+lr[:,2,:,:]*0.114).unsqueeze(1)
                        last_lr_Y = (lr_temporal_sequence[-2][:,0,:,:]*0.299+lr_temporal_sequence[-2][:,1,:,:]*0.587+lr_temporal_sequence[-2][:,2,:,:]*0.114).unsqueeze(1)
 
                        curr_Y_avg_pool = F.avg_pool2d(curr_lr_Y,11,1,5)
                        curr_Y_avg_pool = F.avg_pool2d(curr_Y_avg_pool,11,1,5)

                        last_Y_avg_pool = F.avg_pool2d(last_lr_Y,11,1,5)
                        last_Y_avg_pool = F.avg_pool2d(last_Y_avg_pool,11,1,5)

                        ECG_feature = curr_Y_avg_pool-last_Y_avg_pool 

                        if iter==7:
                            if Extreme_anomaly_flag==0:
                                sr,memory_bank,optical_flow,Ec_lr = net.forward_EC(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,ECG_feature,DEVICE,is_train=True)
                                memory_temporal_sequence.append(memory_bank)
                                flow_temporal_sequence.append(optical_flow)
                                loss_pixel += criterion_pixelwise(hr, sr)
                                pixel_loss_count += 1

                                loss_EC += criterion_pixelwise(lr_original, Ec_lr)
                                ec_loss_count += 1
                                lr_temporal_sequence = lr_temporal_sequence[:-1]
                                lr_temporal_sequence.append(Ec_lr)
                            else:
                                sr,memory_bank,optical_flow,Ec_lr = net.forward_EC(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,ECG_feature,DEVICE,is_train=True)
                                memory_temporal_sequence.append(memory_bank)
                                flow_temporal_sequence.append(optical_flow)
                                AE_flag = 1 

                                loss_EC += criterion_pixelwise(lr_original, Ec_lr)
                                ec_loss_count += 1
                                loss_pixel += criterion_pixelwise(hr, sr)
                                pixel_loss_count += 1


                        elif AE_flag:
                            AE_flag=0
                            memory_gate = [0,1,1]
                            sr,memory_bank,optical_flow = net.forward_converge(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,DEVICE,1,memory_gate,is_train=True)
                            memory_temporal_sequence.append(memory_bank)
                            flow_temporal_sequence.append(optical_flow)
                            loss_pixel += criterion_pixelwise(hr, sr)
                            pixel_loss_count += 1 


                        elif iter==9:
                            memory_gate = [1,0,1]
                            sr,memory_bank,optical_flow = net.forward_converge(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,DEVICE,0,memory_gate,is_train=True)
                            memory_temporal_sequence.append(memory_bank)
                            flow_temporal_sequence.append(optical_flow)
                            loss_pixel += criterion_pixelwise(hr, sr)
                            pixel_loss_count += 1


                        elif iter%3==0:
                            memory_gate = [1,1,1]
                            sr,memory_bank,optical_flow = net.forward_converge(lr,lr_temporal_sequence,memory_bank,memory_temporal_sequence,flow_temporal_sequence,DEVICE,0,memory_gate,is_train=True)
                            memory_temporal_sequence.append(memory_bank)
                            flow_temporal_sequence.append(optical_flow)
                            loss_pixel += criterion_pixelwise(hr, sr)
                            pixel_loss_count += 1
                        else:
                            sr,memory_bank,optical_flow = net.forward(lr,lr_temporal_sequence[-2],memory_bank)
                            memory_temporal_sequence.append(memory_bank)
                            flow_temporal_sequence.append(optical_flow)
                            loss_pixel += criterion_pixelwise(hr, sr)
                            pixel_loss_count += 1


                optimizer_G.zero_grad()
                loss_pixel = loss_pixel/pixel_loss_count
                loss_EC = loss_EC/ec_loss_count
                loss_all = loss_pixel+cfg.Lambda_2*loss_EC
                loss_all.backward()
                optimizer_G.step()
                scheduler_G.step()
                
                print('current epoch: %d | current clip: %s,%s,%s,%s | learn_rate: %4f | current frame: %d | current count: %d | Pixel_loss: %4f | EC_loss: %4f | OUE_flag: %d | EA_flag: %d'%(epoch_count+1,train_file_name_list[clip_index*4],train_file_name_list[clip_index*4+1],train_file_name_list[clip_index*4+2],train_file_name_list[clip_index*4+3],optimizer_G.param_groups[0]['lr'], iter+1,count_iter,cfg.Lambda_1 * loss_pixel.detach().item(),cfg.Lambda_2*loss_EC.detach().item(),OUE_flag,Extreme_anomaly_flag))                
                del lr_temporal_sequence
                del loss_pixel
                del loss_EC
                del loss_all
                
                if epoch_count+1 <= 4500: 
                    if count_iter<5000:
                        if (count_iter) % 1000 == 0:
                            torch.save(net.state_dict(),os.path.join('./save_model',cfg.experiment_index,'MVSR_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                            evolution(cfg,'REDS4',net,int(count_iter),[valloader_000,valloader_011,valloader_015,valloader_020],DEVICE)
                            torch.cuda.empty_cache() 
                    else:
                        if (count_iter) % 5000 == 0:
                            torch.save(net.state_dict(),os.path.join('./save_model',cfg.experiment_index,'MVSR_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                            evolution(cfg,'REDS4',net,int(count_iter),[valloader_000,valloader_011,valloader_015,valloader_020],DEVICE)
                            torch.cuda.empty_cache() 
                else:
                    if (count_iter) % 300 == 0:
                        torch.save(net.state_dict(),os.path.join('./save_model',cfg.experiment_index,'MVSR_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                        evolution(cfg,'REDS4',net,int(count_iter),[valloader_000,valloader_011,valloader_015,valloader_020],DEVICE)
                        torch.cuda.empty_cache()             
        print('run time: ',time.time()-begin_time)        




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
