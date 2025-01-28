import torch
import torch.nn as nn
from torchvision import transforms
import os
class DinoFineTuned(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=2):
        super(DinoFineTuned, self).__init__()
        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.dino_model.head = nn.Linear(self.dino_model.embed_dim, num_classes)
        self.dino_model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device("cpu")))

    def forward(self, x):
        return self.dino_model(x)

def convert_to_onnx(finetuned_model_path, onnx_output_path):
    model = DinoFineTuned(pretrained_model_path=finetuned_model_path)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224) 

    torch.onnx.export(
        model,                      
        dummy_input,                
        onnx_output_path,           
        export_params=True,         
        opset_version=12,          
        do_constant_folding=True,   
        input_names=["input"],      
        output_names=["output"],    
        dynamic_axes={              
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    )
    print(f"Model has been successfully converted to ONNX format at {onnx_output_path}")

if __name__ == "__main__":
    finetuned_model_path = "finetuned_dinov2_vits14.pth"  
    onnx_output_path = "finetuned_dinov2_vits14.onnx"

    if not os.path.exists(finetuned_model_path):
        print(f"Fine-tuned model file not found: {finetuned_model_path}")
    else:
        convert_to_onnx(finetuned_model_path, onnx_output_path)
