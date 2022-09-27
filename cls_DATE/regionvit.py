
from statistics import mode
from threading import local
from turtle import forward

import torch
from torch import nn,einsum
import torch.nn.functional as F

from einops import rearrange,repeat
from einops.layers.torch import Rearrange




class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    
    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs)+x
    
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
    #    project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self,x,mask=None):
        b,n,_,h = *x.shape,self.heads
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t: rearrange(t,'b n (h d) -> b h n d',h=h),qkv)
        
        dots = einsum(' b h i d,b h j d -> b h i j',q,k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j,b h j d -> b h i d',attn,v)

        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class RegionBlock(nn.Module):
    def __init__(self,dim,heads,dim_head,mlp_dim,dropout=0.):
        super().__init__()
        self.attn1 = PreNorm(dim,Attention(dim,heads,dim_head,dropout))
        self.norm = PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout))
    
    def forward(self,region_x,local_x):
       # rb,rn,rd = region_x.shape
        lb,ln,lp,ld = local_x.shape

        # y_r = x_r + RSA(LN(x_r))
        region_x = region_x + self.attn1(region_x)


        # y = y_r || x_l
        region_cls_token = region_x[:,0,:]
        region_data_x = region_x[:,1:,:]
        
        region_x = region_data_x.unsqueeze(2)
        
    
        regional_with_x_locals = torch.cat((region_x,local_x),dim=2)
        regional_with_x_locals = regional_with_x_locals.permute(1,0,2,3).contiguous().reshape(ln*lb,lp+1,ld)

        # LSA: z = y+LSA(LN(y))
        regional_with_x_locals = regional_with_x_locals + self.attn1(regional_with_x_locals)

        # FFN: x = z + FFN(ln(z))
        regional_with_x_locals = regional_with_x_locals + self.norm(regional_with_x_locals)

        regional_with_x_locals = regional_with_x_locals.view(ln,lb,lp+1,ld).permute(1,0,2,3).contiguous()

        region_data,local_x = torch.split(regional_with_x_locals,[1,lp],dim=2)
        
        region_x = region_data.squeeze(2)
        region_x = torch.cat((region_cls_token.unsqueeze(1),region_x),dim=1)

        return region_x,local_x



class RegionTransformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,num_channel,num_patches,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                RegionBlock(dim=dim,heads=heads,dim_head=dim_head,mlp_dim=mlp_dim,dropout=dropout)
            )
        
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(
                nn.Conv2d(num_channel,num_channel,[1,2],1,0)
            )
            
    
    def forward(self,region_x,local_x,mask=None):
        
        last_output_region = []
        n1 = 0
        for attn in self.layers:
            last_output_region.append(region_x)
               
            if n1 > 1:
                #region
                cur_cls_token = region_x[:,0:1]
                cur_data_token = region_x[:,1:]
                last_cls_token = last_output_region[n1-2][:,0:1]
                last_data_token = last_output_region[n1-2][:,1:]

                #region
                cal_q = cur_cls_token
                cal_q_last = last_cls_token
                #region 
                cal_qkv = torch.cat([cal_q,last_data_token],dim=1)
                cal_qkv_last = torch.cat([cal_q_last,cur_data_token],dim=1)

                temp = torch.cat([cal_qkv.unsqueeze(3),cal_qkv_last.unsqueeze(3)],dim=3) 

                region_x = self.skipcat[n1-2](temp).squeeze(3)
            #    print(temp_local.shape)
            region_x,local_x = attn(region_x,local_x)
            n1 += 1
        return region_x,local_x

# num_regions: number of region
# num_patchs : number of local patch for each region
    
class ViT(nn.Module):
    def __init__(self,image_size,near_band,num_regions,num_patches,num_classes,dim,depth,heads,mlp_dim, dim_head = 16, dropout=0., emb_dropout=0.):
        super().__init__()
        patch_dim = image_size ** 2 * near_band 
       
        dim_local = patch_dim // num_patches 
       
        if patch_dim %num_patches != 0:
            dim_local += 1
     #   num_channel = num_regions+1
        num_channel=num_regions+1
        self.pos_embedding = nn.Parameter(torch.randn(1,num_regions+1+num_patches*num_regions,dim))
    #    self.pos_embedding = nn.Parameter(torch.randn(1,31,dim))
        self.patch_to_embedding = nn.Linear(patch_dim,dim)
        self.patch_to_embedding_local = nn.Linear(dim_local,dim)

        self.cls_token = nn.Parameter(torch.randn(1,1,dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = RegionTransformer(dim,depth,heads,dim_head,mlp_dim,num_channel,num_patches,dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes)
        )

    def forward(self,x):
    
        region_x = x
        b,d,p,p = region_x.shape
        region_x = region_x.reshape(b,d,-1)
    #    num_regions = 1
    # for indian num_patches= 7
    # for SA and PU num_patches= 3
        num_patches =7
        local_x = ProductPatchForRegion(region_x,num_patches)
       # print(local_x.shape)

        # region 
        x = self.patch_to_embedding(region_x)
        b,n,_ = x.shape

        # local
        local_x = self.patch_to_embedding_local(local_x)  # b * num_region * num_patch* dim
    #    print(local_x.shape)
        lb,ln,lp,dim_local = local_x.shape

        local_x = local_x.permute(0,2,1,3).contiguous().view(lb,lp*ln,dim_local)

        cls_token = repeat(self.cls_token,'() n d -> b n d',b=b)  # [b,1,dim]
        x = torch.cat((cls_token,x),dim=1)
        x = torch.cat((x,local_x),dim=1)
        x += self.pos_embedding[:,:]
        x = self.dropout(x)

        x,local_x = torch.split(x,(n+1,lp*ln),dim=1)
        local_x = local_x.view(lb,lp,ln,dim_local).permute(0,2,1,3).contiguous()

        x,local_x = self.transformer(x,local_x)
        x = x[:,0] 
    #    x = self.to_latent(x[:,0])
        x = self.mlp_head(x)
        return x
def DataPreHandle(x,num_regions):
    b,n,d = x.shape
 #   print(x.shape)
    kk = n % num_regions
    if  kk != 0:
 #       print(kk)
        pad = torch.zeros(b,num_regions-kk,d).cuda()
 #       print(pad.shape)
        x = torch.cat((x,pad),dim=1)
 #   print(x.shape)
    b,n,d = x.shape
    region_x = x.reshape(b,n//num_regions,-1)
 #   print("opeowww")
 #   print(region_x.shape)
    return region_x


# product patch for each region
# region_size : 每个region块中每片patch大小
def ProductPatchForRegion(x,num_patches):
    # print("duaaa")
    # print(x.shape)
    
    b,n,d = x.shape
    y = d % num_patches
    if y != 0:
        x = F.pad(x,pad=(0,num_patches-y),mode="constant",value=0)
    
    # print(x.shape)

#    num = d // size_patches
    local_x = x.reshape(b,n,num_patches,-1)
    #print(local_x.shape)
    return local_x
