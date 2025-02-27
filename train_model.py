import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Create models with explicit Input layers
def create_user_model():
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8)
    )
    return model

def create_policy_model():
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8)
    )
    return model

try:
    # Create new models
    user_model = create_user_model()
    policy_model = create_policy_model()
    
    # Save models
    torch.save(user_model.state_dict(), 'user_model.pth')
    torch.save(policy_model.state_dict(), 'policy_model.pth')
    
    # Verify loading
    test_user = create_user_model()
    test_user.load_state_dict(torch.load('user_model.pth'))
    test_policy = create_policy_model()
    test_policy.load_state_dict(torch.load('policy_model.pth'))
    print("Models saved and verified successfully!")
    
except Exception as e:
    print(f"Error: {e}") 