import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearnerModule(nn.Module):

    def __init__(self, in_channels, summerize_num_tokens, num_groups, dropout_rate):

        super(TokenLearnerModule, self).__init__()
        self.in_channels = in_channels
        self.summerize_num_tokens = summerize_num_tokens

        self.num_groups = num_groups

        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.GELU(),
            nn.Conv1d(self.in_channels, self.summerize_num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )


        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        selected = inputs
        selected = self.norm(selected) 
        selected = selected.permute(0, 2, 1)  
        selected = self.attention_maps(selected) 
        selected = F.softmax(selected, dim=-1) 


        feat = inputs
        feat = feat.permute(0, 2, 1) 
        feat = self.feat_conv(feat) 
        feat = self.gelu(feat) 
        feat = feat.permute(0, 2, 1) 

        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
       
        outputs = self.dropout(outputs)

        return outputs


class TokenLearnerModuleV11(nn.Module):

    
    def __init__(self, in_channels, summerize_num_tokens, num_groups, dropout_rate):

        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.summerize_num_tokens = summerize_num_tokens
        
        self.num_groups = num_groups
        
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(nn.Linear(self.in_channels, self.in_channels),
                                            nn.GELU(),
                                            nn.Linear(self.in_channels, self.summerize_num_tokens))
        
        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, inputs):

        selected = inputs
        selected = self.norm(selected)
        
        selected = selected.permute(0, 2, 1)
        
        selected = selected.transpose(1, 2)
        selected = self.attention_maps(selected)
        
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, dim=-1)

        
        feat = inputs
        
        feat = feat.permute(0, 2, 1)
        
        feat = self.feat_conv(feat)
        feat = self.gelu(feat)
        
        feat = feat.permute(0, 2, 1)
        
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs
    
class TokenLearnerModuleV12(nn.Module):

    def __init__(self, in_tokens, summerize_num_tokens, num_groups, dropout_rate, dim):

        super(TokenLearnerModuleV12, self).__init__()
        self.in_tokens = in_tokens
        self.summerize_num_tokens = summerize_num_tokens

        self.num_groups = num_groups

        self.norm = nn.LayerNorm(self.in_tokens)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(in_tokens, in_tokens, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.GELU(),
            nn.Conv1d(self.in_tokens, self.summerize_num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        self.feat_conv = nn.Conv1d(
            self.in_tokens, dim, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        selected = inputs
        
        selected = self.norm(selected) 
        selected = selected.permute(0, 2, 1)
        selected = self.attention_maps(selected) 
        selected = F.softmax(selected, dim=-1) 

        feat = inputs

        feat = self.feat_conv(feat) 
        feat = self.gelu(feat) 

        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs