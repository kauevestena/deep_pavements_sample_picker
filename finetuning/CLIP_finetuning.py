from finetuning_lib import *
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from clip.model import CLIP
from clip import clip

EPOCHS = 100
MODEL_NAME = "ViT-L/14@336px" #"ViT-B/32"
TRAIN_PERC = 0.6
DATA_ROOTPATH = SURFACE_SAMPLES_ROOTPATH
EXTRA_PART_NAME = '_default_configs_'
RESIZE = 336

# other params:
norm_means = (0.5, 0.5, 0.5)
norm_stds = (0.5, 0.5, 0.5)
batch_size = 32
lr = 1e-4

print('finetuning on the following classes: ',*get_available_dataset_classes())

metadata = {
    'classes': get_available_dataset_classes_numbers()
}

# Load pre-trained CLIP model
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

# configs dict
configs = {
    'norm_means': norm_means,
    'norm_stds': norm_stds,
    'resize': RESIZE,
    'batch_size': batch_size,
    'lr': lr
}

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(norm_means, norm_stds),
])

# Define dataset
full_dataset = datasets.ImageFolder(root=DATA_ROOTPATH, transform=transform)

# Split dataset into training and testing sets
train_size = int(TRAIN_PERC * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Define dataloaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Fine-tune the model
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

accuracies = []
lossess = []

# Training loop
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for images, labels in tqdm(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        logits_per_image, _ = model(images)
        loss = criterion(logits_per_image, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.cuda()
            labels = labels.cuda()

            logits_per_image, _ = model(images)
            _, predicted_labels = torch.max(logits_per_image, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    accuracies.append(accuracy)
    lossess.append(loss.item())
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}, Test Accuracy: {accuracy}")

# adding info to the model metadata
metadata['epochs'] = epoch+1
metadata['train_perc'] = TRAIN_PERC
metadata['accuracy'] = accuracies
metadata['loss'] = lossess
metadata['configs'] = configs
metadata['data_split'] = recover_samples(full_dataset,train_dataset, test_dataset)

# Save the fine-tuned model
modelname = MODEL_NAME+EXTRA_PART_NAME+'finetuned'

## saving model metadata:
outpath_md = os.path.join(FINETUNING_ROOTPATH,modelname+'.json')
dump_json(metadata, outpath_md)

## saving the model
outpath = os.path.join(FINETUNING_ROOTPATH,modelname+'.pth')
torch.save(model.state_dict(), "fine_tuned_clip_model.pth")
