
import torchvision.models as models

import torch
import torch.nn as nn
#from mmaction.models.backbones.swin_transformer import SwinTransformer3D

from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, PositionalEncodingPermute1D
import torch.nn.functional as F

from einops import rearrange
import numpy as np

from S4Block import S4Decoder
from S4Block import S4DecoderPool

from RepNetModel import *

from S4Block import S4Decoder
from transformers import SwinModel



class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        #out = self.bn(out)
        out = F.softmax(-out/13.544, dim = -1)
        return out

# Take frame encodings, just put them directly through a transformer
class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()
    
        self.frame_encoder = models.resnet50(pretrained=True)
        #for param in self.frame_encoder.parameters():
        #    param.requires_grad = False

        self.frame_encoder.fc = nn.Identity()
        self.temp = 13.544
        self.pos_encode = PositionalEncoding1D(512)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.reduce = nn.Sequential(
            nn.Linear(2048, 512),
        )

        #period length prediction
        self.count = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 32)
        )

    def forward(self, V):

        bs, ch, f, h, w = V.shape
        #print("V shape is", V.shape)

        x = rearrange(V, 'bs ch f h w -> (bs f) ch h w')
        x = self.frame_encoder(x)
        #x = self.reduce(x)
        #x = F.normalize(x)

        x = rearrange(x, '(bs f) ch -> bs f ch', bs=bs, f=f)
        x = F.normalize(self.reduce(x))
        x = x + self.pos_encode(x)
        x = self.trans(x)

        x = self.count(x)

        #print(x.shape, self.counting_tensor.shape)
        #print(x)
        #print(x)
        return x, 1

# Take frame encodings, just put them directly through S4 Decoder
class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
    
        self.frame_encoder = models.resnet50(pretrained=True)
        #for param in self.frame_encoder.parameters():
        #    param.requires_grad = False

        self.frame_encoder.fc = nn.Identity()
        self.temp = 13.544
        self.pos_encode = PositionalEncoding1D(512)

        self.S4 = [S4Decoder().cuda() for i in range(3)]

        self.reduce = nn.Sequential(
            nn.Linear(2048, 512),
        )

        #period length prediction
        self.count = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 32)
        )

    def forward(self, V):

        bs, ch, f, h, w = V.shape
        #print("V shape is", V.shape)

        x = rearrange(V, 'bs ch f h w -> (bs f) ch h w')
        x = self.frame_encoder(x)
        #x = self.reduce(x)
        #x = F.normalize(x)

        x = rearrange(x, '(bs f) ch -> bs f ch', bs=bs, f=f)
        x = F.normalize(self.reduce(x))
        x = x + self.pos_encode(x)
        for layer in self.S4:
            x = layer(x)

        x = self.count(x)

        #print(x.shape, self.counting_tensor.shape)
        #print(x)
        #print(x)
        return x, 1
        
# S4 Decoder followed by self-similarity matrix
class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
    
        self.frame_encoder = models.resnet50(pretrained=True)
        #for param in self.frame_encoder.parameters():
        #    param.requires_grad = False

        self.frame_encoder.fc = nn.Identity()
        self.temp = 13.544
        self.pos_encode = PositionalEncoding1D(512)

        self.S4 = [S4Decoder().cuda() for i in range(3)]

        self.sims = Sims()
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        self.bn1 = nn.BatchNorm3d(512)
        self.bn2 = nn.BatchNorm2d(32)

        self.reduce = nn.Sequential(
            nn.Linear(2048, 512),
        )
        self.input_projection = nn.Linear(64 * 32, 512)

        #period length prediction
        self.count = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 32)
        )

    def forward(self, V):

        bs, ch, f, h, w = V.shape
        #print("V shape is", V.shape)

        x = rearrange(V, 'bs ch f h w -> (bs f) ch h w')
        x = self.frame_encoder(x)
        #x = self.reduce(x)
        #x = F.normalize(x)

        x = rearrange(x, '(bs f) ch -> bs f ch', bs=bs, f=f)
        x = F.normalize(self.reduce(x))
        x = x + self.pos_encode(x)
        for layer in self.S4:
            x = layer(x)

        x = F.relu(self.sims(x))
        x = F.relu(self.bn2(self.conv3x3(x)))
        x = F.dropout(x, p=0.25) 
        x = rearrange(x, 'bs ch f1 f2 -> bs f1 (f2 ch)')    

        x = self.count(x)

        #print(x.shape, self.counting_tensor.shape)
        #print(x)
        #print(x)
        return x, 1


class S4Enrichment1(nn.Module):
    def __init__(self):
        super(S4Enrichment1, self).__init__()

        self.s4decoder = S4Decoder(in_feats=1024, out_feats=512)
        self.posenc = PositionalEncoding3D(1024)


    def forward(self, x):
        
        bs, ch, f, h, w = x.shape
        #print("here is the size of x", x.shape)
        x = rearrange(x, "bs ch f h w -> bs f h w ch")
        x = x + self.posenc(x)
        x = rearrange(x, 'bs f h w ch -> bs (f h w) ch')
        x = self.s4decoder(x) 
        x = rearrange(x, 'bs (f h w) ch -> bs ch f h w', f=f, h=h, w=w)


        return x

# Exact same arch as RepNet, but replace the temporal enrichment with S4 blocks 
class Network4(nn.Module):
    def __init__(self):
        super(Network4, self).__init__()
    
        self.repnet = RepNet(num_frames=64)
        self.repnet.conv3D = S4Enrichment1()


    def forward(self, x):
        return self.repnet(x)

class Network5(nn.Module):
    def __init__(self, num_frames=64):
        super(Network5, self).__init__()

        self.num_frames = num_frames
        self.backbone = models.resnet50(pretrained=True, progress=True)
        self.backbone.fc = nn.Identity()
        self.S41 = S4Decoder(in_feats=2048, out_feats=1024)
        self.S42 = S4Decoder(in_feats=1024, out_feats=512)
    
        self.sims = Sims()
        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, 32)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)


    def forward(self, x):
        bs, ch, f, h, w = x.shape

        x = rearrange(x, 'bs ch f h w -> (bs f) ch h w')
        x = self.backbone(x)
        x = rearrange(x, '(bs f) ch -> bs f ch', bs=bs, f=f)
        x = F.relu(self.bn1(self.S42(self.S41(x)).permute(0,2,1)).permute(0,2,1))        

        x = F.relu(self.sims(x))
        xret = x
        
        x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame
        #print(x.shape)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bs, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        
        #period
        x1 = self.transEncoder1(x)
        y1 = x1.transpose(0, 1)
        y1 = F.relu(self.ln1_2(self.fc1_1(y1)))
        y1 = F.relu(self.fc1_2(y1))
        #y1 = F.relu(self.fc1_3(y1))

        return y1, 1

# Just a video encoder
class Network6(nn.Module):
    def __init__(self):
        super(Network6, self).__init__()
    
        self.backbone = models.video.r2plus1d_18(pretrained=True)
        self.backbone.fc = nn.Identity()
    
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, 32)


    def forward(self, x):
        bs, ch, f, h, w = x.shape
        x = self.backbone(x)
        x = self.fc1_2(self.ln1_2(self.fc1_1(x)))
        x = x.unsqueeze(1).repeat(1, 64, 1)
        return x, 1

class Network7(nn.Module):
    def __init__(self):
        super(Network7, self).__init__()
        #self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        self.s41 = S4DecoderPool(in_feats=2048, out_feats=1024)
        self.s42 = S4DecoderPool(in_feats=1024, out_feats=512)
        self.s43 = S4DecoderPool(in_feats=512, out_feats=256, kernel_size=(1,1,1), stride=(1,1,1))

        self.do1 = nn.Dropout(p=0.1)
        self.ln = nn.LayerNorm(2048)

        self.num_frames = 64

        # Just repnet stuff now
        self.sims = Sims()
        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, 32)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)

    def forward(self, x):
        bs, ch, f, h, w = x.shape
        x = rearrange(x, 'bs ch f h w-> (bs f) ch h w')
        x = self.backbone(x)
        x = rearrange(x, '(bs f) ch h w -> (bs f) h w ch', bs=bs, f=f)
        x = self.do1(x)
        x = self.ln(x)
        x = rearrange(x, '(bs f) h w ch-> bs f h w ch', bs=bs, f=f, h=4, w=4)
        x = self.s41(x)
        x = self.s42(x)
        x = self.s43(x).squeeze(2).squeeze(2)

        # Just repnet
        x = self.sims(x)
        xret = x
        
        x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame
        #print(x.shape)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bs, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        
        #period
        x1 = self.transEncoder1(x)
        y1 = x1.transpose(0, 1)
        y1 = F.relu(self.ln1_2(self.fc1_1(y1)))
        y1 = F.relu(self.fc1_2(y1))
        #y1 = F.relu(self.fc1_3(y1))

        return y1, 1

        return x, 1

if __name__=="__main__":
    network7 = Network7().to('cuda')
    x = torch.rand(3, 3, 64, 112, 112).cuda()
    x, _ = network7(x)
    #print("The now shape of x is", x.shape)
    print(x.shape)