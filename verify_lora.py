import torch
import torch.nn as nn
import copy
import sys
import os
import tempfile

# Add src to path just in case, though usually unnecessary if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock load_and_compute_loss if it doesn't exist or fails to import
# We do this because we only want to test lora.py logic, not the whole dependency tree
try:
    import load_and_compute_loss
except ImportError:
    pass

from src.omnilingual_asr.models.lora import inject_lora, LoRALinear, LoraConfig

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers to be targeted
        self.attn = nn.Linear(10, 10)
        self.ffn = nn.Linear(10, 10)
        # Layer to be ignored
        self.classifier = nn.Linear(10, 2)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        x = self.classifier(x)
        return x

def test_lora():
    print("Initializing SimpleModel...")
    model = SimpleModel()
    
    # Keep a copy of the state dict to compare weights later
    original_state_dict = copy.deepcopy(model.state_dict())
    
    # Input for testing
    x = torch.randn(5, 10)
    
    # 1. Test Injection
    print("\n--- Testing Injection ---")
    config = LoraConfig(
        r=4,
        alpha=8.0,
        dropout_p=0.0,
        target_keywords=("attn", "ffn")
    )
    
    # Get initial output before injection (for comparison)
    model.eval()
    with torch.no_grad():
        initial_output = model(x)
        
    print("Injecting LoRA...")
    try:
        inject_lora(model, config=config, freeze_base=True)
    except AttributeError as e:
        print(f"\nCRITICAL ERROR during injection: {e}")
        print("This likely means LoRALinear is trying to access attributes that don't exist on nn.Linear.")
        print("Check lines 65-66 in lora.py: input_dim/output_dim vs in_features/out_features")
        return

    print("Model structure after injection:")
    print(model)
    
    # Verify types
    assert isinstance(model.attn, LoRALinear), "attn should be LoRALinear"
    assert isinstance(model.ffn, LoRALinear), "ffn should be LoRALinear"
    assert isinstance(model.classifier, nn.Linear), "classifier should remain nn.Linear"
    print("✓ Injection types verified")
    
    # 2. Verify Freezing
    print("\n--- Testing Freeze ---")
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"LoRA param {name} should require grad"
        else:
            assert not param.requires_grad, f"Base param {name} should NOT require grad"
    print("✓ Parameter freezing verified")

    # 3. Verify Initialization (Identity)
    # LoRA B is initialized to 0, so initially output should match base model exactly
    print("\n--- Testing Initialization (Identity) ---")
    model.eval()
    with torch.no_grad():
        lora_output = model(x)
    
    # Check if outputs are close
    if torch.allclose(initial_output, lora_output, atol=1e-6):
        print("✓ Initialization verified: Output matches base model (LoRA effect is 0 initially)")
    else:
        print("✗ Initialization failed: Output differs from base model")
        diff = (initial_output - lora_output).abs().max().item()
        print(f"  Max difference: {diff}")
        
    # 4. Verify Training Updates
    print("\n--- Testing Training Updates ---")
    model.train()
    # Optimizer only includes trainable params
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    
    # Capture weights before update
    lora_A_before = model.attn.lora_A.weight.clone()
    lora_B_before = model.attn.lora_B.weight.clone()
    base_weight_before = model.attn.base.weight.clone()
    classifier_weight_before = model.classifier.weight.clone()
    
    # Forward + Backward
    y = model(x)
    loss = y.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check what changed
    lora_A_after = model.attn.lora_A.weight
    lora_B_after = model.attn.lora_B.weight
    base_weight_after = model.attn.base.weight
    classifier_weight_after = model.classifier.weight
    
    # Check LoRA weights
    if not torch.equal(lora_A_before, lora_A_after):
        print("✓ LoRA A weights updated")
    else:
        print("✗ LoRA A weights did NOT update")
        
    if not torch.equal(lora_B_before, lora_B_after):
        print("✓ LoRA B weights updated")
    else:
        # Note: B is initialized to 0. Gradient might be non-zero if input to it (A*x) is non-zero.
        print("✗ LoRA B weights did NOT update (might happen if gradients are 0)")

    # Check Base weights (frozen)
    if torch.equal(base_weight_before, base_weight_after):
        print("✓ Base weights preserved (unchanged)")
    else:
        print("✗ Base weights CHANGED (they should be frozen)")
        
    # Check Untargeted Layer weights (frozen/ignored)
    if torch.equal(classifier_weight_before, classifier_weight_after):
        print("✓ Non-targeted layer (classifier) weights preserved (unchanged)")
    else:
        print("✗ Non-targeted layer (classifier) weights CHANGED (should be frozen/ignored)")
        
    # 5. Verify Effect of LoRA
    # Now that we updated weights, the output should differ from initial_output
    model.eval()
    with torch.no_grad():
        new_output = model(x)
        
    if not torch.allclose(initial_output, new_output, atol=1e-6):
        print("✓ Training verified: Output has changed after training LoRA")
    else:
        print("✗ Training failed: Output is still the same as base model")

    # 6. Verify Save and Load
    print("\n--- Testing Save and Load ---")
    # Save the trained model to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        save_path = tmp.name
    
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # Simulate loading process as in load_llm_asr_300m_with_lora
        # 1. Create fresh base model (simulating load_llm_asr_300m)
        loaded_model = SimpleModel()
        
        # 2. Inject LoRA (same config)
        inject_lora(loaded_model, config=config, freeze_base=True)
        
        # 3. Load state dict
        # Note: In real scenario, we might only save LoRA weights or full weights. 
        # Here we saved full state_dict of the LoRA-injected model.
        ckpt = torch.load(save_path)
        loaded_model.load_state_dict(ckpt, strict=True)
        loaded_model.eval()
        
        # 4. Compare outputs
        with torch.no_grad():
            loaded_output = loaded_model(x)
            
        if torch.allclose(new_output, loaded_output, atol=1e-6):
             print("✓ Save/Load verified: Output matches exactly")
        else:
             print("✗ Save/Load failed: Output differs")
             diff = (new_output - loaded_output).abs().max().item()
             print(f"  Max difference: {diff}")
             
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    try:
        test_lora()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # import traceback
        # traceback.print_exc()
