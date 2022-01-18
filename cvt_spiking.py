import os
import torch
from SpykeTorch import functional as sf
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


use_cuda = True
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,                 
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        k_conv1 = 5
        k_conv2 = 2
        self.conv1 = snn.Convolution(self.in_channels, self.out_channels, k_conv1, 0.8, 0.05)
        self.conv1_t = 10
        self.k1 = 1
        self.r1 = 2

        self.conv2 = snn.Convolution(self.out_channels, self.out_channels, k_conv2, 0.8, 0.05)
        self.conv2_t = 1
        self.k2 = 1
        self.r2 = 1

        self.stdp1 = snn.STDP(self.conv1, (0.04, -0.03))
        self.stdp2 = snn.STDP(self.conv2, (0.04, -0.03))
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0

    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input):
        # print(self.training,"333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333")
        # print(input.shape, "**************************************8===============================")        
        input = sf.pad(input.float(), (2,2,2,2), 0)
        # print(input.shape, "**************************************8===============================")
        if self.training:
            # print(self.in_channels,"self.in_channels")
            # print(self.out_channels,"out_channels,")            
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            
            self.spk_cnt1 += 1
            if self.spk_cnt1 >= 5000:

                self.spk_cnt1 = 0
                ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                ap = torch.min(ap, self.max_ap)
                an = ap * -0.75
                self.stdp1.update_all_learning_rate(ap.item(), an.item())
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
            self.save_data(input, pot, spk, winners)

            if pot.shape[1] == 512:     
                # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk99999999999999")                       
                x = 2
                y = 50
                z = 1
            if pot.shape[1] == 256:
                # print ("yes...............................yes.....................")
                x = 47
                y = 2
                z = 1


            # print(pot.shape, "pot pot pot ...................................")
                
            spk_in = sf.pad(sf.pooling(spk, y ,y, z), (x,x,x,x))
            # print(spk_in.shape, "spk_in 11111111111111111111111")
            spk_in = sf.pointwise_inhibition(spk_in)
            # print(spk_in.shape, "spk_in 22222222222222222222")            
            pot = self.conv2(spk_in)
            # print(pot.shape, "pot 333333333333333333")
            spk, pot = sf.fire(pot, self.conv2_t, True)
            # print("44444444444444444444444444444")            
            pot = sf.pointwise_inhibition(pot)
            # print("55555555555555555555555555555555555")
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
            self.save_data(spk_in, pot, spk, winners)

            spk_out = sf.pooling(spk, 2, 2, 1)
            if spk_out.shape[2] == 56:
                # print("&&&&&&&&&&&&&&&&&&****************")
                spk_out = sf.pad(sf.pooling(spk, 4 ,4, 1), (2,2,2,2))
            if spk_out.shape[2] == 52:
                # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                spk_out = sf.pad(sf.pooling(spk, 7 ,7, 1), (1,1,1,1))
            # print(spk_out.shape, "spk_outtttttttttttttttttttttttttttttttt")            
            return spk_out
        else:
            # print("else, else ........................................................................" )
            # print(self.in_channels,"self.in_channels")
            # print(self.out_channels,"out_channels,")            
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if pot.shape[1] == 512:     
                # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk99999999999999")                       
                x = 2
                y = 50
                z = 1
            if pot.shape[1] == 256:
                # print ("yes...............................yes.....................")
                x = 47
                y = 2
                z = 1                
            spk_in = sf.pad(sf.pooling(spk, y ,y, z), (x,x,x,x))      
            pot = self.conv2(spk_in)
            # print(pot.shape, "pot 333333333333333333")
            spk, pot = sf.fire(pot, self.conv2_t, True)
            spk_out = sf.pooling(spk, 2, 2, 1)
            if spk_out.shape[2] == 56:
                # print("&&&&&&&&&&&&&&&&&&****************")
                spk_out = sf.pad(sf.pooling(spk, 4 ,4, 1), (2,2,2,2))
            if spk_out.shape[2] == 52:
                # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                spk_out = sf.pad(sf.pooling(spk, 7 ,7, 1), (1,1,1,1))
            # print(spk_out.shape, "spk_outtttttttttttttttttttttttttttttttt")            
            return spk_out
    def stdp(self):
        # print("how are you")
        self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim*16, 256])
        self.norm1 = nn.LayerNorm([dim*4, 256])  
        self.norm2 = nn.LayerNorm([256, 256])  
        self.norm3 = nn.LayerNorm([257, 256])     
        self.fn = fn
    def forward(self, x, **kwargs):
        # print(x.shape, "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        if x.shape[1] == 4096:
            # print ("reallyyyyyyyyyyyyyyy")
            return self.fn(self.norm(x), **kwargs)
        if x.shape[1] == 1024:
            # print ("reallyyyyyyyyyyyyyyy000000000000000000")
            return self.fn(self.norm1(x), **kwargs)
        if x.shape[1] == 256:
            # print ("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            return self.fn(self.norm2(x), **kwargs)
        if x.shape[1] == 257:
            return self.fn(self.norm3(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(x.shape, "forward forward ffffffffffffffffffffff")
        return self.net(x)



class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head ,  dropout = 0.5, last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.dim = dim
        self.img_size = img_size
        self.heads = 1
        self.dim_head = dim_head     
        self.inner_dim = self.dim_head *  self.heads
        project_out = not (heads == 1 and dim_head == self.dim)
        self.scale = self.dim_head ** -0.5
     
        #
        self.to_q = SepConv2d(self.dim, self.inner_dim)
        self.to_k = SepConv2d(self.dim, self.inner_dim)
        self.to_v = SepConv2d(self.dim, self.inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
   
        # print(self.heads , "self.heads ")
        # print(self.dim,"print dim dim")
        # print(self.scale,"self.scale")
        # print(self.inner_dim,"print inner dim dim")  
        # print(x.shape,"x shape x shape x shape ...........................")   
        # print()   
        # if x.shape[1] == 256:
        #     print("yesyesyes")
        #     x = x.reshape(1,256,128)
        b, n, _, h = *x.shape, self.heads  
        # print(b)
        # print(h,"hhhhhhhhhhhhhhhhhhh, what")
        
        if self.last_stage:
            # print(x.shape, "99999999999999999999")
            cls_token = x[:, 0]
            # print(cls_token.shape, "99999999999999999999")            
            x = x[:, 1:]
            # print(x.shape, "99999999999999999999")            
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        # print("helloooooooooooooooooooooo,kkkkkkkkkkkkkkkkkk")     
        # print(x.shape, "99999999999999999999 helloooooooooooooooooooo")       
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)    
        # print(x.shape,"helloooooooooooooooooooooo,lllllllllllllll")      
        # x = x.reshape()
        q = self.to_q(x)
        # print(q.shape,"q shape 0000000000000000000000000000000000000000000000000088888888888888888888888")       
        self.to_q.stdp()       
        # q=torch.stack(list(q), dim=0)

        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        # print(q.shape,"q shape")              
        v = self.to_v(x)
        # print(v.shape,"vvvvvvvvvvshape")
        self.to_v.stdp()        
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(x)
        # print(k.shape,"k k shape  shape")              
        self.to_k.stdp()        
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            # print("yyyyyyyyyyyyyyyyyyyyffffffffffffffffffffhhhhhhhhhhhhhhhhhh")
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        # print(self.scale)
        # print("helloooooooooooooooooooooooonnnnnnnnnnnnnnnnnnyyyyyyyyyyyyy")
        # print(q.shape,k.shape,v.shape,"999999999999999999")
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # print(dots.shape,"kkkkkkkkkkk")
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        # print(out.shape, "helloooooooooooooooooooooooooooooooo,ppppppppjjjjjj")
        return out



class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim,dropout, last_stage=False):                        
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim = dim
        self.img_size = img_size  
        self.dim_head = dim_head          
        self.dropout = dropout       
        self.dim_head = dim_head        
        self.last_stage = last_stage      
        self.mlp_dim = mlp_dim     
        for _ in range(depth):      
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim, ConvAttention(self.dim, self.img_size, self.heads, self.dim_head,self.dropout, self.last_stage)), 
                PreNorm(self.dim, FeedForward(self.dim, self.mlp_dim, self.dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:        
            # print(type(attn(x)),attn(x))
            x = attn(x) + x  	
            x = ff(x) + x
            # print(x.shape,"x-shape,2222222222222222222222222222")	      
        return x



class CvT(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes,s2_kernel_size,
               threshold, stdp_lr, anti_stdp_lr,dropout=0.5, image_size=256, dim=256, kernels=[256, 256, 256], strides=[193, 33, 17],
                 heads=[1, 2,6] , depth = [1, 2, 10],pool='cls', emb_dropout=0.5, scale_dim=4):
        super(CvT, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.kernel_size = s2_kernel_size
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.anti_stdp_lr = anti_stdp_lr
        self.dropout = torch.ones(self.number_of_features) * dropout
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        ##### Stage 1 #######
        self.conv1 = snn.Convolution(input_channels, kernels[0], strides[0], 0.8, 0.05)
        self.stdp1 = snn.STDP(self.conv1, (0.0004, -0.0003))                
        self.conv1_t = 15
        self.k1 = 1
        self.r1 = 3
        self.R1 = Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4)
        self.norm1 = nn.LayerNorm([int(self.dim*16),256])        
        self.stage1_transformer = Transformer(dim=dim, img_size=image_size//4,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
        self.R11 = Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)


        ##### Stage 2 #######

        scale = heads[1]//heads[0]
        self.dim2 = scale*self.dim
        self.conv2 = snn.Convolution(256, kernels[1], strides[1], 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 1
        self.r2 = 1      
        self.stdp2 = snn.STDP(self.conv1, (0.04, -0.03))  
        self.R2 = Rearrange('b c h w -> b (h w) c', h = image_size//8, w = image_size//8)
        self.norm2 = nn.LayerNorm([self.dim2*2,256])
        
        self.stage2_transformer =  Transformer(dim=dim,     img_size=image_size//8, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)                              
                                           
        self.R22 = Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
        
        ##### Stage 3 #######
        input_channels = self.dim2
        scale = heads[2] // heads[1]
        self.dim3 = self.dim
        self.conv3 = snn.Convolution(256, kernels[2], strides[2], 0.8, 0.05)
        self.stdp3 = snn.STDP(self.conv3, (0.04, -0.03), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.04, 0.005), False, 0.2, 0.8)
        self.R3 = Rearrange('b c h w -> b (h w) c', h = image_size//16, w = image_size//16)
        self.norm3 = nn.LayerNorm([self.dim3,256])        
        self.stage3_transformer = Transformer(dim=dim, img_size=image_size//16, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True)
        self.R33 = Rearrange('b (h w) c -> b c h w', h = image_size//16, w = image_size//16)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)


        self.norm33 = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, self.number_of_classes)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map = []
        for i in range(1000):
            self.decision_map.extend([i]*20)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        input = input.float()
        # print(input.shape,"input shape")       
        if self.training:
            pot = self.conv1(input)
            # print(pot.shape, "pot shape pp")  
            pot = self.R1(pot)
            # print(pot.shape, "pot shape r1r1")              
            pot = self.norm1(pot)   
            # print(pot.shape, "pot shape norm1")                         
            pot = self.stage1_transformer(pot)   
            # print(pot.shape, "pot shape stage1")                  
            pot = self.R11(pot)     
            # print(pot.shape, "pot shape R11")                            
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 5000:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            # print(pot.shape, "pot shape kk")   
            spk_in =  spk
            pot = self.conv2(spk_in)
            # print(pot.shape, "conv2")
            pot = self.R2(pot)
            # print(pot.shape, "r2")      
            # print(self.dim2,"print self.dim2")  
            pot = self.norm2(pot)
            # print(pot.shape, "norm2")            
            pot = self.stage2_transformer(pot)   
            pot = self.R22(pot) 
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 5000:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                # print(spk.shape, "spk spk spk lllllllllllllllllllllllllllllllllllllllllllllllllllllll")
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                # print(winners,"00000000000000000000000000000000000000000000222222222222222222999999999999999999999")                
                self.save_data(input, pot, spk, winners)
                return spk, pot

            pot = self.conv3(spk)
            # print(pot.shape,"77777777777777777777777777777777777")
            pot = self.R3(pot)
            # print(self.dim3)
            # print(pot.shape, "666666666666666666666666666666666666666666666666666666666")
            pot = self.norm3(pot)
            b, n, _ = pot.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            pot = torch.cat((cls_tokens, pot), dim=1)
            # print(pot.shape, "88888888888888888888888888888888888888888888888888888")            
            pot = self.stage3_transformer(pot)
            # pot = self.R33(pot)
            # pot = pot.mean(dim=1) if self.pool == 'mean' else pot[:, 0]
            # pot = self.norm33(pot)
            print(pot.shape)
            # pot = self.mlp_head(pot)
            pot = pot.reshape(1,257,16,16)
            spk = sf.fire(pot)
            print(spk.shape, "spk spk spk zzzzzzzzzzzzzzzzzzzzzzzzzz")            
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                print(winners[0][0],"000000000000000000000000000000000000000000002222222222222222229999999999999999999999")
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
#            print(pot.shape,"0000000000000000000000000))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")
            pot = self.R1(pot)
            pot = self.norm1(pot)
            pot= self.stage1_transformer(pot)            
            pot = self.R11(pot)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(spk)
            pot = self.R2(pot)
            pot = self.norm2(pot)
            pot = self.stage2_transformer(pot) 
            pot = self.R22(pot)                           
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(spk)             
            pot = self.R3(pot)
            pot = self.norm3(pot)
            b, n, _ = pot.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            pot = torch.cat((cls_tokens, pot), dim=1)
            pot = self.stage3_transformer(pot)
            # pot = self.R33(pot)
            # pot = pot.mean(dim=1) if self.pool == 'mean' else pot[:, 0]     
            # pot = self.norm33(pot)
            # pot = self.mlp_head(pot)  
            pot = pot.reshape(1,257,16,16)                          
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output


    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

