from finetuning_lib import *


def fine_tune_clip(model, dataset, learning_rate=1e-5, epochs=5, test_split=0.2):
    # Split dataset into training and test sets
    train_size = int(len(dataset) * (1 - test_split))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Don't shuffle test data

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train loop
        model.train()  # Set model to training mode
        for images, labels in train_dataloader:
            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(torch.zeros_like(images))  # Create zero text features for classification

            # Concatenate and project features using a linear layer (adjust as needed)
            combined_features = torch.cat((image_features, text_features), dim=1)
            logits = model.head(combined_features)

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress (optional)
            # print(f"Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")

        # Evaluation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for efficiency
            test_loss = 0
            test_correct = 0
            for images, labels in test_dataloader:
                # Forward pass (similar to train loop)
                # Calculate loss and accuracy (adapt based on your task)
                test_loss += criterion(logits, labels).item()
                test_correct += (logits.argmax(dim=1) == labels).sum().item()

            test_acc = test_correct / len(test_dataset)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save model at the end of each epoch (optional)
        torch.save(model.state_dict(), f"clip_model_{epoch+1}.pt")

    print("Fine-tuning complete!")