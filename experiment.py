import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Triphase Loss Implementation
# ==========================================
class TriphaseLoss(nn.Module):
    """
    Physics-Informed Loss Function based on Triphase Cosmology Theory.
    
    This loss function models the universe (or the learning system) as an interference 
    network of three phases:
    
    1. Positive Phase (H+): 
       Represents the observable reality or 'accuracy'. 
       Implemented as Standard CrossEntropy Loss.
       
    2. Negative Phase (H-): 
       Represents gravitational tension or structural constraints. 
       Implemented as L2 Regularization (Weight Decay) to prevent overfitting.
       Controlled by parameter 'alpha'.
       
    3. Imaginary/Informational Phase (iHim): 
       Represents informational coherence. 
       Implemented as Entropy Minimization to reduce internal uncertainty (decoherence).
       Controlled by parameter 'beta'.
       
    Total Loss = L_pos + (alpha * L_neg) + (beta * L_im)
    """
    def __init__(self, alpha=0.0005, beta=0.1):
        super(TriphaseLoss, self).__init__()
        self.alpha = alpha  # Strength of Negative Phase (Gravity/Tension)
        self.beta = beta    # Strength of Imaginary Phase (Coherence)

    def forward(self, outputs, targets, model):
        # 1. Positive Phase (H+): Projection to reality (Classification Error)
        l_pos = F.cross_entropy(outputs, targets)
        
        # 2. Negative Phase (H-): Structural Tension (L2 Regularization)
        l_neg = 0
        for param in model.parameters():
            l_neg += torch.sum(param ** 2)
            
        # 3. Imaginary Phase (iHim): Informational Coherence (Entropy Minimization)
        # Minimizing entropy drives the system towards a 'coherent' state (C=0).
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        l_im = -(probs * log_probs).sum(dim=1).mean()

        # Calculate Total Interference Tensor (Total Loss)
        total_loss = l_pos + (self.alpha * l_neg) + (self.beta * l_im)
        
        return total_loss, l_pos.item(), l_neg.item(), l_im.item()

# ==========================================
# 2. Model Definition: CNN
# ==========================================
class TriphaseCNN(nn.Module):
    """
    Standard Convolutional Neural Network (CNN) for MNIST classification.
    Used to demonstrate the efficacy of Triphase Loss under noisy conditions.
    """
    def __init__(self):
        super(TriphaseCNN, self).__init__()
        # Conv Layer 1: 1ch -> 16ch, kernel 3x3
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        # Conv Layer 2: 16ch -> 32ch, kernel 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # FC Layers
        # 28x28 -> conv1(26x26) -> conv2(24x24) -> maxpool(12x12) -> 32ch * 12 * 12 = 4608
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ==========================================
# 3. Experiment Engine
# ==========================================
def run_experiment(alpha, beta, epochs=8, noise_level=1.2):
    """
    Runs a training and evaluation loop.
    Injects Gaussian noise into the training data to simulate a high-entropy environment.
    """
    print(f"Running Experiment: alpha={alpha}, beta={beta}, noise={noise_level} ...")
    
    # Data Preparation: Inject heavy noise to test coherence capabilities
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + noise_level * torch.randn_like(x)) 
    ])
    
    # Test data remains clean (measuring generalization capability)
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    # Download and load MNIST data
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # Initialize Model and Optimizer
    model = TriphaseCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = TriphaseLoss(alpha=alpha, beta=beta)

    history = {'train_loss': [], 'iHim': [], 'test_acc': []}

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0
        ihim_sum = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss using Triphase logic
            loss, l_pos, l_neg, l_im = criterion(output, target, model)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += l_pos
            ihim_sum += l_im
        
        # Evaluation Loop
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        
        # Logging
        avg_loss = loss_sum / len(train_loader)
        avg_ihim = ihim_sum / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['iHim'].append(avg_ihim)
        history['test_acc'].append(acc)
        
        print(f"  Epoch {epoch}/{epochs}: Train Loss={avg_loss:.4f}, iHim={avg_ihim:.4f}, Test Acc={acc:.2f}%")

    return history

# ==========================================
# 4. Main Execution: Parameter Sweep
# ==========================================
if __name__ == '__main__':
    # Define experiment parameters
    # Compare Baseline (Standard SGD) vs Triphase Optimization
    params = [
        {'alpha': 0.0, 'beta': 0.0, 'label': 'Baseline (Normal)'},
        {'alpha': 0.0005, 'beta': 0.2, 'label': 'Triphase (Balanced)'},
        {'alpha': 0.005, 'beta': 0.5, 'label': 'Triphase (Strong)'},
    ]
    
    results = {}
    EPOCHS = 8
    NOISE_LEVEL = 1.2
    
    print("=== Triphase CNN Sweep Experiment Start ===")
    
    # Run experiments
    for p in params:
        res = run_experiment(p['alpha'], p['beta'], epochs=EPOCHS, noise_level=NOISE_LEVEL)
        results[p['label']] = res

    # Generate and Save Plots
    print("Generating graphs...")
    epochs_range = range(1, EPOCHS + 1)
    
    # Plot 1: Test Accuracy
    plt.figure(figsize=(10, 6))
    for label, res in results.items():
        plt.plot(epochs_range, res['test_acc'], marker='o', label=label)
    plt.title(f'Test Accuracy Comparison (CNN + Gaussian Noise {NOISE_LEVEL})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_accuracy.png')
    
    # Plot 2: Information Phase Coherence (iHim)
    plt.figure(figsize=(10, 6))
    for label, res in results.items():
        plt.plot(epochs_range, res['iHim'], marker='s', linestyle='--', label=label)
    plt.title('Information Phase Coherence (Entropy Minimization)')
    plt.xlabel('Epochs')
    plt.ylabel('iHim Value (Lower is Better)')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_coherence.png')

    print("\n=== Experiment Complete! ===")
    print("Results saved: result_accuracy.png, result_coherence.png")