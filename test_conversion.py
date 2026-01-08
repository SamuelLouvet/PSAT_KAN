import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from converted_KAN.kan_converter import KanConverter

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model, loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    print(f"--- Evaluating {name} ---")
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of {name}: {accuracy:.2f} %')
    return accuracy

def main():
    BATCH_SIZE = 128
    
    # 1. Load Data (CIFAR-10 Test Set only)
    print("Preparing Data...")
    
    # ResNet models typically expect specific normalization
    # Using standard CIFAR-10 mean/std
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # 2. Load Pre-trained Model
    print("\n=== Loading Pre-trained ResNet20 ===")
    # Using the same repo as the notebook
    repo = "chenyaofo/pytorch-cifar-models"
    model_name = "cifar10_resnet20"
    
    try:
        model_orig = torch.hub.load(repo, model_name, pretrained=True, verbose=True)
        model_orig = model_orig.to(device)
    except Exception as e:
        print(f"Error loading model from hub: {e}")
        return

    print("Original Model loaded.")
    # Verify original accuracy
    acc_orig = evaluate_model(model_orig, testloader, name="Original ResNet20")

    # 3. Convert to KAN
    print("\n=== Converting to KAN ===")
    import copy
    model_to_convert = copy.deepcopy(model_orig)
    
    converter = KanConverter()
    model_kan = converter.convert_model(model_to_convert)
    model_kan = model_kan.to(device)
    
    print("\nConverted Model Architecture:")
    print(model_kan)
    
    # 4. Evaluate KAN
    acc_kan = evaluate_model(model_kan, testloader, name="Converted ResNet20 (KAN)")

    # 5. Comparison
    print("\n=== Comparison ===")
    print(f"Original Model Accuracy: {acc_orig:.2f}%")
    print(f"KAN Model Accuracy:      {acc_kan:.2f}%")
    diff = acc_kan - acc_orig
    print(f"Difference:              {diff:+.2f}%")
    
    if abs(diff) < 0.1:
        print("\nSUCCESS: The converted model matches the original performance.")
    else:
        print("\nWARNING: Discrepancy detected.")

if __name__ == "__main__":
    main()
