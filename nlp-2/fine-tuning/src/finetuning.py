import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer:nn.Linear, r=4, alpha=32):
        super().__init__()
        self.r:int = r
        self.alpha: int = alpha
        self.original_layer: nn.Module = original_layer

        self.A:torch.Tensor = torch.zeros((original_layer.out_features, r))
        self.B: torch.Tensor = torch.zeros((r, original_layer.in_features))

        nn.init.kaiming_uniform_(self.A, a = 5**(1/2))
        
        self.scaling = alpha/r

        for param in original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x:torch.Tensor):
        original_out: torch.Tensor = self.original_layer(x)
        LoRa_out: torch.Tensor = x.matmul(self.A@self.B)
        
        return original_out + self.scaling * LoRa_out

def inject_lora_into_model(model:nn.Module, r=4, alpha=32, device='cpu'):
    verify_names:set = set()
    
    for child_name, child_module in model.named_children():
        
        if type(child_module) == nn.Linear:
            verify_names.add(child_name)
            
        if child_name.lower() in ["o", "v", "k", "q"]:
            lora_layer = LoRA(child_module, r, alpha)
            setattr(model, child_name, lora_layer)
            
        else:
            inject_lora_into_model(child_module, r, alpha, device)
    
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        super().__init__()
        self.soft_prompt:nn.Parameter = nn.Parameter(torch.zeros((prompt_length, model_hidden_size)))
        nn.init.kaiming_normal_(self.soft_prompt)

    def forward(self, input_embeddings: torch.Tensor):
        batch_size:int = input_embeddings.shape[0]
        soft_prompt_expanded:torch.Tensor = torch.stack([self.soft_prompt for _ in range(batch_size)])

        return torch.cat([soft_prompt_expanded, input_embeddings], 1)