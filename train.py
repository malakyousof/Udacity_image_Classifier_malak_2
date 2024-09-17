import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import argparse
import json
from collections import OrderedDict
import os

# Argument Parser
parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images')
parser.add_argument('data_dir', type=str, help='Path to the dataset directory (mandatory)')
parser.add_argument('--save_dir', type=str, default='cd0673/43ef1733-cf74-40a9-8c22-b6e440128200/image-classifier-part-1-workspace/home/aipnd-project', help='Directory to save the trained model checkpoint')
parser.add_argument('--arch', type=str, default='vgg13', choices=['vgg13', 'densenet121'], help='Model architecture (vgg13 or densenet121)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')  # Added argument for resuming
args = parser.parse_args()

# Define directories for datasets
train_dir = os.path.join(args.data_dir, 'train')
valid_dir = os.path.join(args.data_dir, 'valid')
test_dir = os.path.join(args.data_dir, 'test')

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# Load class-to-name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Model Selection
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_size = 25088
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024
else:
    raise ValueError("Unsupported architecture. Choose vgg13 or densenet121.")

# Freeze pre-trained model parameters
for param in model.parameters():
    param.requires_grad = False

# Define new classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, args.hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(args.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Set device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Load checkpoint if specified
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epochs']
        model.class_to_idx = checkpoint['class_to_idx']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at '{args.resume}'")

# Validation function
def validate_model(model, loader, criterion):
    model.eval()
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return loss / len(loader), accuracy / len(loader)

# Training loop
epochs = args.epochs
steps = 0
print_every = 10
running_loss = 0

for epoch in range(start_epoch, epochs):
    model.train()
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss:.3f}.. "
                  f"Validation accuracy: {valid_accuracy:.3f}")

            running_loss = 0
            model.train()

    # Save the checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_train10.pth')
        model.class_to_idx = train_dataset.class_to_idx
        checkpoint = {
            'arch': args.arch,
            'class_to_idx': model.class_to_idx,
            'model_state_dict': model.state_dict(),
            'classifier': model.classifier,
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epoch + 1,
            'learning_rate': args.learning_rate
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path} after epoch {epoch+1}")

# Final Testing
test_loss, test_accuracy = validate_model(model, test_loader, criterion)
print(f"Test accuracy: {test_accuracy:.3f}")

# Save the final checkpoint
final_checkpoint_path = os.path.join(args.save_dir, 'final_checkpoint.pth')
model.class_to_idx = train_dataset.class_to_idx
final_checkpoint = {
    'arch': args.arch,
    'class_to_idx': model.class_to_idx,
    'model_state_dict': model.state_dict(),
    'classifier': model.classifier,
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs,
    'learning_rate': args.learning_rate
}

torch.save(final_checkpoint, final_checkpoint_path)
print(f"Final model saved at {final_checkpoint_path}")


   