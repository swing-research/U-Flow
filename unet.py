import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unet_utils import *

class encoder_unet(nn.Module):
    # U-Net encoder
    def __init__(self, n_channels,init_features, res):
      super(encoder_unet, self).__init__()
    
      self.n_channels = n_channels
      self.init_features = init_features
      self.res = res
      down_net = []
      down_net.append(DoubleConv(n_channels, self.init_features))
      in_ch = self.init_features
    
      for i in range(int(math.log(self.res//2,4))):
        out_ch =  2 *in_ch
        down_net.append(Down(in_ch, out_ch))
        in_ch = out_ch
      
      if self.res == 64 or self.res == 256:
        out_ch= 2* in_ch
        down_net.append(Downi(in_ch, out_ch))
      else:
        in_ch=in_ch//2

      self.down_net = nn.ModuleList(down_net)

    def forward(self, x):
      resi = []
      for i in range(len(self.down_net)):
        x = self.down_net[i](x)
        if i != len(self.down_net)-1:
          resi.append(x)

      return x, resi



class decoder_unet(nn.Module):
  # U-Net decoder
  def __init__(self, n_classes,init_features, res):
    super(decoder_unet, self).__init__()
    self.n_classes = n_classes
    self.init_features = init_features
    self.res = res
    up_net = []

    if self.res == 64 or self.res ==256:
      out_ch = 2**(int(math.log(self.res//2,4))+1)* init_features
      in_ch = out_ch//2
    else:
      out_ch =2**(int(math.log(self.res//2,4))) * init_features
      in_ch = out_ch//2

    if self.res == 64 or self.res == 256:
      up_net.append(Up(out_ch, in_ch))
      out_ch = in_ch
      in_ch = in_ch//2
        
    for k in range(int(math.log(self.res//2,4))):
      up_net.append(Upi(out_ch, in_ch))
      out_ch = in_ch
      in_ch = in_ch//2
        
    up_net.append(OutConv(self.init_features, n_classes))

    self.up_net = nn.ModuleList(up_net)

  def forward(self, x,resi):
      
    for i in range(len(self.up_net)-1):
      x = self.up_net[i](x,resi[len(resi)-i-1])

    logits = self.up_net[-1](x)
    return logits



class UNet(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(UNet, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
       
    def forward(self, x,resi = None, dir = 'encoder'):
    
        if dir == 'encoder':
            x , resi = self.encoder(x)
            return x , resi

        elif dir == 'decoder':
            x = self.decoder(x, resi)
            return x
        

