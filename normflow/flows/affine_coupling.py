import numpy as np
import torch
from torch import nn

from .base import Flow
from .reshape import Split, Merge



class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('s', torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('t', torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        #print('salam')
        #print(z.size())
        #print(self.t.size())
        #print(self.s.size())
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """

    def __init__(self, shape, num_classes):
        super().__init__()
        self.shape = shape
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.s_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.t_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = z * torch.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = (z - t) * torch.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

def squeeze(z,f):
    s = z.size()
    z = z.view(*s[:2], s[2] // f, f, s[3] // f, f)
    z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
    z = z.view(s[0], f*f * s[1], s[2] // f, s[3] // f)
    
    return z

class condition_net(nn.Module):

    def __init__(self, c_in = 3,c_out = 2,factor = 2, res_in = 256, res_out = 32):

        super(condition_net, self).__init__()

        self.factor =  factor
        res_out = res_out//factor
        self.factori =  res_in//res_out 
        
        self.conv1 = nn.Conv2d(self.factori*self.factori*c_in, 16 ,padding= 'same' ,kernel_size=3, stride = 1)
        self.conv2 = nn.Conv2d(16, 32,padding= 'same', kernel_size=3, stride = 1)
        self.conv3 = nn.Conv2d(32, self.factor*c_out,padding= 'same', kernel_size=3, stride = 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = squeeze(x,self.factori)
        x = self.relu(self.conv1(x))
        x = self.conv3(self.relu(self.conv2(x)))
        return self.sigmoid(x)



class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map='exp', c_in= 3, c_out= 2 , factor = 2,res_in = 256, res_out = 32):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid
        scale when sampling from the model
        """
        super().__init__()
        self.add_module('param_map', param_map)
        self.scale = scale
        self.scale_map = scale_map
        self.cond = condition_net(c_in= c_in,c_out = c_out, factor = factor, res_in = res_in, res_out = res_out)

    def forward(self, z,y):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """

        z1, z2 = z

        y_hat = self.cond(y)
        #z1 = z1+z1_hat
        z1_hat = torch.cat((z1,y_hat), dim = 1)
        param = self.param_map(z1_hat)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale),
                                     dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale),
                                    dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z,y):
        z1, z2 = z
        #print(y.size())
        #print(z1.size())
        #print(y_hat.size())
        y_hat = self.cond(y)
        #print(z1.size())
        #print(y_hat.size())
        z1_hat = torch.cat((z1,y_hat), dim = 1)
       
        #print(z1_hat.size())

        param = self.param_map(z1_hat)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale),
                                    dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale),
                                     dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 -= param
            log_det = 0
        #print('salam')
        #print(z1.size())
        #print(z2.size())
        return [z1, z2], log_det


class MaskedAffineFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    Masked affine flow f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """
        Constructor
        :param b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        :param t: translation mapping, i.e. neural network, where first input dimension is batch dim,
        if None no translation is applied
        :param s: scale mapping, i.e. neural network, where first input dimension is batch dim,
        if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer('b', self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module('s', s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module('t', t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """
    def __init__(self, param_map, scale=True, scale_map='exp', split_mode='channel', c_in = 3,c_out = 2 , factor = 2, res_in= 256, res_out = 32):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.factor=factor
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map, c_in= c_in, c_out = c_out , factor = self.factor, res_in = res_in, res_out = res_out)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z,y):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z,y)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z,y):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            #print(z.size())
            z, log_det = self.flows[i].inverse(z,y)
            log_det_tot += log_det
        return z, log_det_tot