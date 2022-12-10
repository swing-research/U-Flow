import torch
import normflow as nf

def real_nvp(latent_dim, K=64):

    
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_dim)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        t = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_dim)]
    
    q0 = nf.distributions.DiagGaussian(latent_dim)
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    
    return nfm




def glow(L=3, K=16, hidden_channels = 256 , c_in = 3,c_out = 2, res_in = 256, res_out = 32):

    input_shape = (c_out , res_out, res_out)
    #n_dims = np.prod(input_shape)
    channels = c_out
    split_mode = 'channel'
    scale = True
    # num_classes = 10

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                        split_mode=split_mode, scale=scale, scale_map = 'sigmoid_inv', c_in = c_in,c_out = c_out, factor = 2**(L-i), res_in = res_in, res_out = res_out)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                        input_shape[2] // 2 ** (L - i))
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        # q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]
        q0 += [nf.distributions.DiagGaussian(latent_shape)]

    # Construct flow model
    nfm = nf.MultiscaleFlow(q0, flows, merges , class_cond=False)
        
    
    return nfm
