"""
Debug script to inspect model structure and find correct attribute names for REAP observer.
Run this in Kaggle to see the actual MoE module attributes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "ai-sage/GigaChat3-10B-A1.8B-bf16"

print(f"Loading config for {model_id}...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print("\n" + "="*60)
print("CONFIG ATTRIBUTES")
print("="*60)
for attr in dir(config):
    if not attr.startswith('_'):
        val = getattr(config, attr, None)
        if not callable(val) and not isinstance(val, dict):
            print(f"  {attr}: {val}")

print("\n" + "="*60)
print("Loading model (this may take a while)...")
print("="*60)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"\nModel class: {model.__class__.__name__}")
print(f"Number of layers: {len(model.model.layers)}")

# Find MoE module in first MoE layer (usually layer 1 or later)
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        mlp_class = mlp.__class__.__name__
        print(f"\nLayer {layer_idx} MLP class: {mlp_class}")
        
        if 'MoE' in mlp_class or 'Moe' in mlp_class:
            print(f"\n" + "="*60)
            print(f"FOUND MoE MODULE at layer {layer_idx}: {mlp_class}")
            print("="*60)
            
            print("\nMoE module attributes:")
            for attr in dir(mlp):
                if not attr.startswith('_'):
                    val = getattr(mlp, attr, None)
                    if not callable(val):
                        if isinstance(val, (int, float, str, bool, type(None))):
                            print(f"  {attr}: {val}")
                        elif isinstance(val, torch.nn.ModuleList):
                            print(f"  {attr}: ModuleList with {len(val)} modules")
                        elif isinstance(val, torch.nn.Module):
                            print(f"  {attr}: {val.__class__.__name__}")
                        elif hasattr(val, 'shape'):
                            print(f"  {attr}: Tensor {val.shape}")
            
            # Check if it has config
            if hasattr(mlp, 'config'):
                print("\n  mlp.config attributes:")
                for attr in dir(mlp.config):
                    if not attr.startswith('_'):
                        val = getattr(mlp.config, attr, None)
                        if not callable(val) and not isinstance(val, dict):
                            print(f"    config.{attr}: {val}")
            
            # Check experts
            if hasattr(mlp, 'experts'):
                print(f"\n  Number of experts: {len(mlp.experts)}")
                if len(mlp.experts) > 0:
                    expert = mlp.experts[0]
                    print(f"  Expert class: {expert.__class__.__name__}")
                    print(f"  Expert attributes: {[a for a in dir(expert) if not a.startswith('_') and not callable(getattr(expert, a, None))]}")
            
            # Check router/gate
            if hasattr(mlp, 'gate'):
                print(f"\n  Router/gate class: {mlp.gate.__class__.__name__}")
                print(f"  Router attributes: {[a for a in dir(mlp.gate) if not a.startswith('_') and not callable(getattr(mlp.gate, a, None))]}")
            
            break
    
    if layer_idx > 5:  # Only check first few layers
        break

print("\n" + "="*60)
print("DONE - Use the attributes above to configure REAP observer")
print("="*60)
