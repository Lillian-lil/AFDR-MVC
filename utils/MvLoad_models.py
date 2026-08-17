import torch
from multi_vae.MvModels import VAE

def load(view_num=2,
         Network='C',
         hid=256,
         img_size=(1, 32, 32),
         path='./example-model.pt',
         latent_spec={"disc": [10], "cont": 10},
         shareAE=1,
         use_cuda=False,
         num_heads=4):
    """
    Load a trained model, compatible with both old and new model formats.
    """
    path_to_model = path
    
    # Create a new model instance
    model = VAE(latent_spec=latent_spec, img_size=img_size, view_num=view_num,
                Network=Network, hidden_dim=hid, shareAE=shareAE, num_heads=num_heads)
    model.missing_view_handling = True
      
    device = torch.device('cuda' if use_cuda else 'cpu')
       
    state_dict = torch.load(path_to_model, map_location=device)
   
    # Check if the saved model is an old version (without mi_estimators)
    is_old_model = True
    for key in state_dict.keys():
        if "mi_estimators" in key:
            is_old_model = False
            break
    
    if is_old_model:
        new_state_dict = model.state_dict()
        migrated_keys = 0
        # Transfer all matching parameters from the old model
        for key, value in state_dict.items():
            if key in new_state_dict:
                new_state_dict[key] = value
                migrated_keys += 1
        
        # Initialize newly added mi_estimators with default values
        for name, param in model.named_parameters():
            if "mi_estimators" in name:
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
        
        state_dict = new_state_dict
        print(f"Successfully migrated {migrated_keys} parameters to the new model architecture")
    
    # Load state dict (allow partial mismatch)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Print loading results
    print(f"Model loaded on device: {device}")
    if missing_keys:
        print(f"Warning: Missing keys - using default initialization:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys:")
        for key in unexpected_keys:
            print(f"  - {key}")
    
    return model