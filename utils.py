import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import numpy as np



def conditional_sampling(nfm, unet, x_test , y_test ,device,squeeze , f, n_average , n_test = 5 , n_sample_show = 4):
    '''Generate posterior samples, MMSE and UQ'''

    def normalization(image):
        image += -image.min()
        image /= (image.max())
        
        return image
    
    y_s_single = y_test[0:n_test]
    y_reshaped = y_test[0:n_test,0,:,:].detach().cpu().numpy()

    y_s = torch.zeros(n_test * n_average,y_test.size()[1],y_test.size()[2],y_test.size()[3]).to(device)

    for j in range(n_test):

        xx = y_s_single[j]
        xx = xx.expand(1,-1,-1,-1)

        y_s[j*n_average:j*n_average+n_average] = xx.repeat(n_average, 1, 1, 1)
      
    gt = x_test[0:n_test].detach().cpu().numpy()

    _, resi = unet(y_s, 'encoder')

    z_random_base,_ = nfm.sample(y_s, torch.tensor(n_average * n_test).to(device))

    z_random_basei = squeeze(z_random_base,f)
    x_sampled = unet(z_random_basei, resi, 'decoder').detach().cpu().numpy()

    n_sample = n_sample_show + 4
    final_shape = [n_test*(n_sample), np.shape(x_sampled)[2] , np.shape(x_sampled)[3],
                   np.shape(x_sampled)[1]]
    x_sampled_all = np.zeros(final_shape)
    mean_vec = np.zeros([n_test , np.shape(x_sampled)[2] , np.shape(x_sampled)[3],
                         np.shape(x_sampled)[1]] , dtype = np.float32)
    image_size = x_test.size()[2]
    c = x_test.size()[1]
    
    gt = np.reshape(gt,
                    [gt.shape[0],
                    c,image_size, image_size]).transpose(0,2,3,1)
    
    x_sampled = np.reshape(x_sampled,
                    [x_sampled.shape[0],
                    c,image_size, image_size]).transpose(0,2,3,1)
    
    y_reshaped = np.reshape(y_reshaped,
                    [y_reshaped.shape[0],
                    c,image_size, image_size]).transpose(0,2,3,1)
    y_reshaped = ((y_reshaped - y_reshaped.min())/(y_reshaped.max() - y_reshaped.min())) * 0.5

    SSIM_MMSE = 0

    for i in range(n_test):

        x_sampled_all[i*n_sample] = gt[i]
        post_mean = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        x_sampled_all[i*n_sample + 1] = post_mean
        x_sampled_all[i*n_sample+2:i*n_sample + 2 + n_sample_show] = x_sampled[i*n_average:i*n_average + n_sample_show]
        post_std = np.std(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        post_uq = post_std/(post_mean + 0.3)
        x_sampled_all[i*n_sample + 2 + n_sample_show] = (0.5 * normalization(post_uq))
        x_sampled_all[i*n_sample + 3 + n_sample_show] = y_reshaped[i]
        
        mean_vec[i] = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)

        SSIM_MMSE = SSIM_MMSE + ssim(mean_vec[i],
                    gt[i],
                    data_range=gt[i].max() - gt[i].min(),
                    multichannel=True)

    snr_MMSE = PSNR(gt , mean_vec)
        
    return x_sampled_all , snr_MMSE, SSIM_MMSE/n_test


def PSNR(x_true , x_pred):

    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_pred[i],
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    
def flags():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs_unet',
        type=int,
        default=150,
        help='number of epochs to train autoencoder network')
     
    
    parser.add_argument(
        '--epochs_flow',
        type=int,
        default=150,
        help='number of epochs to train flow network')

    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch_size')
    
    
    parser.add_argument(
        '--dataset', 
        type=str,
        default='scattering',
        help='which dataset to work with')
    
    
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=1,
        help='GPU number')

    parser.add_argument(
        '--remove_all',
        type= int,
        default= 0,
        help='Remove the privious experiment if exists')


    parser.add_argument(
        '--desc',
        type=str,
        default='Default',
        help='add a small descriptor to autoencoder experiment')


    parser.add_argument(
        '--res',
        type=int,
        default=128,
        help='Resolution of the dataset')
    
    parser.add_argument(
        '--channel',
        type=int,
        default=1,
        help='Channel of the dataset')
    
    
    parser.add_argument(
        '--train_unet',
        type=int,
        default=1,
        help='Train autoencoder network')


    parser.add_argument(
        '--train_flow',
        type=int,
        default=1,
        help='Train normalizing flow network')
    
    parser.add_argument(
        '--restore_flow',
        type=int,
        default=1,
        help='Restore the trained flow if exists')

    parser.add_argument(
        '--input_type',
        type=str,
        default='bp',
        help='limited-view or full')
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
