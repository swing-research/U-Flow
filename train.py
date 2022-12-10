import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
import shutil
from flow_model import real_nvp , glow
import normflow as nf
from utils import *
from unet import encoder_unet, decoder_unet, UNet
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader import scattering_dataloader, general_dataloader

torch.manual_seed(0)
np.random.seed(0)

FLAGS, unparsed = flags()
epochs_flow = FLAGS.epochs_flow
epochs_unet = FLAGS.epochs_unet
batch_size = FLAGS.batch_size
dataset = FLAGS.dataset
gpu_num = FLAGS.gpu_num
desc = FLAGS.desc
image_size = FLAGS.res
c = FLAGS.channel
remove_all = bool(FLAGS.remove_all)
train_unet = bool(FLAGS.train_unet)
train_flow = bool(FLAGS.train_flow)
restore_flow = bool(FLAGS.restore_flow)
input_type = FLAGS.input_type

enable_cuda = True
device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

all_experiments = 'experiments/'
if os.path.exists(all_experiments) == False:
    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'unet_' + dataset + '_' \
    + str(image_size) + '_' + desc

if os.path.exists(exp_path) == True and remove_all == True:
    shutil.rmtree(exp_path)

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


data_folder = 'dataset/'
if input_type == 'full':
    address = data_folder + 'full'

elif input_type == 'limited-view':
    address = data_folder + 'limited-view'



if dataset == 'scattering':

    gt_train  = scattering_dataloader(address+'/gt/', typei='gt')
    gt_test  = scattering_dataloader(address+'/gt_test/', typei='gt')

    measure_train  = scattering_dataloader(address + '/y/',typei = 'bp')
    measure_test  = scattering_dataloader(address + '/y_test/',typei = 'bp')



else:

    train_dataset = Dataset_loader(dataset = dataset ,size = (image_size,image_size), c = c)
 


if dataset =='scattering':

    train_loader_gt = torch.utils.data.DataLoader(gt_train, batch_size=batch_size,
                                           num_workers=32)
    train_loader_measure = torch.utils.data.DataLoader(measure_train, batch_size=batch_size,
                                           num_workers=32)

    test_loader_gt = torch.utils.data.DataLoader(gt_test, batch_size=batch_size,
                                           num_workers=32)
    test_loader_measure = torch.utils.data.DataLoader(measure_test, batch_size=batch_size,
                                           num_workers=32)

else:

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=32)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler,
                                           num_workers=32)

ntrain = len(gt_train)

learning_rate = 1e-4
step_size = 50
gamma = 0.5
myloss = F.mse_loss
plot_per_num_epoch = 1 if ntrain > 20000 else 20000//ntrain

# Print the experiment setup:
print('Experiment setup:')
print('---> epochs_flow: {}'.format(epochs_flow))
print('---> batch_size: {}'.format(batch_size))
print('---> dataset: {}'.format(dataset))
print('---> Learning rate: {}'.format(learning_rate))
print('---> experiment path: {}'.format(exp_path))
print('---> image size: {}'.format(image_size))
print('---> Number of training samples: {}'.format(ntrain))


# 1. Training unet:

enc = encoder_unet(n_channels = c ,init_features = 32, res = image_size).to(device)
dec = decoder_unet(n_classes = 1 ,init_features = 32, res = image_size).to(device)
unet = UNet(encoder= enc, decoder = dec).to(device)


test_images = next(iter(test_loader_measure)).to(device)
# test_images = test_images.repeat(1,1,8,8)

embed, _ = unet(test_images,'encoder')
#unet = UNet(n_channels=c, n_classes=c, init_features = 32, res = image_size).to(device)

num_param_unet= count_parameters(unet)
print('---> Number of trainable parameters of unet: {}'.format(num_param_unet))

optimizer_unet = Adam(unet.parameters(), lr=learning_rate)
scheduler_unet = torch.optim.lr_scheduler.StepLR(optimizer_unet, step_size=step_size, gamma=gamma)

checkpoint_unet_path = os.path.join(exp_path, 'unet.pt')
if os.path.exists(checkpoint_unet_path):
    checkpoint_unet = torch.load(checkpoint_unet_path)
    unet.load_state_dict(checkpoint_unet['model_state_dict'])
    optimizer_unet.load_state_dict(checkpoint_unet['optimizer_state_dict'])
    print('unet is restored...')


if train_unet:

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = epochs_unet + 1 # only plot in the last epoch
    
    loss_unet_plot = np.zeros([epochs_unet])
    for ep in range(epochs_unet):
        unet.train()
        t1 = default_timer()
        loss_unet_epoch = 0

        for measure,gt in zip(train_loader_measure, train_loader_gt):

            batch_size = measure.shape[0]
            measure = measure.to(device)
            gt = gt.to(device)

            optimizer_unet.zero_grad()
            embed, resi = unet(measure,dir = 'encoder')
            image_recon = unet(embed,resi, dir = 'decoder')
            #image_recon = unet(yy)

            recon_loss = myloss(image_recon.reshape(batch_size, -1) , gt.reshape(batch_size, -1) )
            #regularization = myloss(embed, torch.zeros(embed.shape).to(device))
            unet_loss = recon_loss #+ regularization

            unet_loss.backward()
    
            optimizer_unet.step()
            loss_unet_epoch += unet_loss.item()


        scheduler_unet.step()

        t2 = default_timer()

        loss_unet_epoch/= ntrain
        loss_unet_plot[ep] = loss_unet_epoch
        
        plt.plot(np.arange(epochs_unet)[:ep], loss_unet_plot[:ep], 'o-', linewidth=2)
        plt.title('unet_loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE loss')

        plt.savefig(os.path.join(exp_path, 'unet_loss.jpg'))
        np.save(os.path.join(exp_path, 'unet_loss.npy'), loss_unet_plot[:ep])
        plt.close()
        
        torch.save({
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer_unet.state_dict()
                    }, checkpoint_unet_path)


        samples_folder = os.path.join(exp_path, 'Generated_samples')
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions_unet')
    
        if not os.path.exists(image_path_reconstructions):
            os.mkdir(image_path_reconstructions)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_unet:
            unet.eval()
           
            sample_number = 4
            ngrid = int(np.sqrt(sample_number))

            test_images_measure = next(iter(test_loader_measure)).to(device)
            test_images_measure = test_images_measure[:sample_number]
            # test_images_measure = test_images_measure.repeat(1,1,8,8)

            test_images_gt = next(iter(test_loader_gt)).to(device)
            test_images_gt = test_images_gt[:sample_number]

            embedi, resii = unet(test_images_measure,dir = 'encoder')
            generated_embed = unet(embedi,resii, dir = 'decoder')

            vv = test_images_measure.detach().cpu().numpy()

            lor_res = torch.zeros(sample_number,1,image_size,image_size)
            low_res = vv[:,0,:,:]
            low_res = np.reshape(low_res,
                                [low_res.shape[0],
                                1,image_size, image_size]).transpose(0,2,3,1)
            low_res = low_res[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, 1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)


            generated_samples = generated_embed.detach().cpu().numpy()
            image_recon_out = generated_samples.transpose(0,2,3,1)
            generated_samples = np.reshape(generated_samples,
                                           [generated_samples.shape[0],
                                            1,image_size, image_size]).transpose(0,2,3,1)

            generated_samples = generated_samples[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, 1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)

            test_images = test_images_gt.detach().cpu().numpy()
            image_np = test_images.transpose(0,2,3,1)
            test_images = np.reshape(test_images,
                                [test_images.shape[0],
                                1,image_size, image_size]).transpose(0,2,3,1)
            test_images = test_images[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, 1).swapaxes(1, 2).reshape(ngrid*image_size, -1, 1)

            #cv2.imwrite(os.path.join(image_path_reconstructions, 'epoch %d_output.png' % (ep,)), generated_samples) # training images
            #cv2.imwrite(os.path.join(image_path_reconstructions, 'epoch %d_input.png' % (ep,)), low_res) # training images
            #cv2.imwrite(os.path.join(image_path_reconstructions, 'epoch %d_gt.png' % (ep,)), test_images) # training images
            plt.imsave(os.path.join(image_path_reconstructions, 'epoch %d_output.png' % (ep,)) ,generated_samples[:,:,0] , cmap='seismic')
            plt.imsave(os.path.join(image_path_reconstructions, 'epoch %d_input.png' % (ep,)) ,low_res[:,:,0] , cmap='seismic')
            plt.imsave(os.path.join(image_path_reconstructions, 'epoch %d_gt.png' % (ep,)) ,test_images[:,:,0] , cmap='seismic')
            

            psnr_unet = PSNR(image_np , image_recon_out)


            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                        file.write('ep: %03d/%03d | time: %.0f | unet_loss %.6f | PSNR_unet  %.1f' %(ep, epochs_unet,t2-t1,
                            loss_unet_epoch, psnr_unet))
                        file.write('\n')

            print('ep: %03d/%03d | time: %.0f | unet_loss %.6f | PSNR_unet  %.1f' %(ep, epochs_unet,t2-t1,
                            loss_unet_epoch, psnr_unet))
        
def squeeze(z,f):
    s = z.size()
    z = z.view(*s[:2], s[2] // f, f, s[3] // f, f)
    z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
    z = z.view(s[0], f*f * s[1], s[2] // f, s[3] // f)
    
    return z

def squeeze_inverse(z,f):

    s = z.size()
    z = z.view(s[0], s[1] // (f*f), f, f, s[2], s[3])
    z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
    z = z.view(s[0], s[1] // (f*f), f * s[2], f * s[3])

    return z

# Training the flow model
if train_flow:
    unet.eval()

    if embed.size()[1]==128:
        f= 8
    else:
        f=16

    em = squeeze_inverse(embed, f)
    c_out , res = em.size()[1], em.size()[2]

    nfm = glow(L=3, K=8, hidden_channels = 256 , c_in= c, c_out =c_out , res_in = image_size, res_out = res)    

    nfm = nfm.to(device)
    num_param_nfm = count_parameters(nfm)
    print('Number of trainable parametrs of flow: {}'.format(num_param_nfm))
    
    loss_hist = np.array([])
    optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=step_size, gamma=gamma)
    
    # Initialize ActNorm
    batch_img = next(iter(train_loader_measure)).to(device)
    embed_batch_img , _ = unet(batch_img,'encoder')
    print(embed_batch_img.shape)
    embed_batch_img = squeeze_inverse(embed_batch_img,f)

    likelihood = nfm.log_prob(embed_batch_img, batch_img)
    
    checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
    if os.path.exists(checkpoint_flow_path) and restore_flow == True:
        checkpoint_flow = torch.load(checkpoint_flow_path)
        nfm.load_state_dict(checkpoint_flow['model_state_dict'])
        optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
        print('Flow model is restored...')
    
    for ep in range(epochs_flow):

        nfm.train()
        t1 = default_timer()
        loss_flow_epoch = 0
        for measure , gt in zip(train_loader_measure, train_loader_gt):
            optimizer_flow.zero_grad()

            measure = measure.to(device)
            embed_batch , _ = unet(measure,'encoder')
            embed_batch = squeeze_inverse(embed_batch,f)

            # Compute loss
            loss_flow = nfm.forward_kld(embed_batch,measure)
            
            if ~(torch.isnan(loss_flow) | torch.isinf(loss_flow)):
                loss_flow.backward()
                optimizer_flow.step()
            
            # Make layers Lipschitz continuous
            #nf.utils.update_lipschitz(nfm, 5)
            
            loss_flow_epoch += loss_flow.item()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss_flow.to('cpu').data.numpy())
        
        scheduler_flow.step()
        t2 = default_timer()
        loss_flow_epoch /= ntrain
        
        torch.save({
                    'model_state_dict': nfm.state_dict(),
                    'optimizer_state_dict': optimizer_flow.state_dict()
                    }, checkpoint_flow_path)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_flow:

            nfm.eval()

            samples_folder = os.path.join(exp_path, 'Generated_samples')
            if not os.path.exists(samples_folder):
                os.mkdir(samples_folder)
            image_path_generated = os.path.join(
                samples_folder, 'generated')
        
            if not os.path.exists(image_path_generated):
                os.mkdir(image_path_generated)


            n_test = 5 # Number of test samples
            n_sample_show = 3 # Number of posterior samples to show for each test sample
            n_average = 25

            test_images = next(iter(test_loader_gt)).to(device)
            test_images = test_images[:n_average]
            yy_test = next(iter(test_loader_measure)).to(device)
            # yy_test = yy_test.repeat(1,1,8,8)
            yy_test = yy_test[:n_average]
            print(test_images.min() , test_images.max())

            x_sampled_conditional, psnr_MMSE , SSIM_MMSE = conditional_sampling(nfm, unet, test_images , yy_test ,device,squeeze, f, n_average , n_test  ,n_sample_show)

            #cv2.imwrite(os.path.join(image_path_generated, 'conditional_sampled_epoch %d.png' % (ep,)),
                #                x_sampled_conditional[:, :, :, ::-1].reshape(
               #         n_test, n_sample_show + 4,
              #          image_size, image_size, 1).swapaxes(1, 2)
             #           .reshape(n_test*image_size, -1, 1)*255.0)
            dd=x_sampled_conditional[:, :, :, ::-1].reshape(n_test, n_sample_show + 4,
                image_size, image_size, 1).swapaxes(1, 2).reshape(n_test*image_size, -1, 1)

            np.save(os.path.join(image_path_generated, 'conditional_sampled_epoch %d.npy' % (ep,)), dd)
            
            plt.imsave(os.path.join(image_path_generated, 'conditional_sampled_epoch %d.png' % (ep,)),
                 dd[:,:,0], cmap='seismic')
        
            
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                    file.write('ep: %03d/%03d | time: %.4f | ML_loss %.4f | PSNR_MMSE %.4f | SSIM_MMSE %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch, psnr_MMSE,SSIM_MMSE))
                    file.write('\n')
    
            print('ep: %03d/%03d | time: %.4f | ML_loss %.4f | PSNR_MMSE %.4f| SSIM_MMSE %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch,psnr_MMSE,SSIM_MMSE))

    
    