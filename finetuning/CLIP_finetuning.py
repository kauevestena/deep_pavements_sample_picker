from finetuning_lib import *

# use your own data
# list_image_path = []
# list_txt = []

images_split_path = os.path.join(FINETUNING_ROOTPATH,MODEL_OUTNAME+'_images_split.json')

# list_image_path, list_txt = simple_class_listing(False)

list_image_path, list_txt, list_image_path_test, list_txt_test = splitted_class_listing(outpath=images_split_path)

dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) #Define your own dataloader

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             betas=BETAS,
                             eps=EPS,
                             weight_decay=WEIGHT_DECAY,
                             amsgrad=AMSGRAD,
                             lr=LR
                            ) # the lr is smaller, more safe for fine tuning to new dataset


# original configs:

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# v5 was trained with:

# optimizer = torch.optim.Adam(model.parameters(),
#                             #  lr=5e-5,
#                              betas=(0.9,0.98),
#                              eps=1e-6,
#                             #  weight_decay=0.2,
#                              weight_decay=0.5,
#                              amsgrad=True,
#                              lr=5e-6
#                             )


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

losses = []

# Train the model
for epoch in tqdm(range(NUM_EPOCHS)):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch 
        
        images= images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()

        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {total_loss.item():.4f}")

        losses.append(total_loss.item())

training_report = {
    'epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'pretrained_model': pretrained_key,
    'model': base_model,
    'model_outname': MODEL_OUTNAME,
    'losses': losses,
}

dump_json(training_report, os.path.join(FINETUNING_ROOTPATH,MODEL_OUTNAME+'_training_report.json'))

model_outpath = os.path.join(FINETUNING_ROOTPATH,MODEL_OUTNAME+'.pt')


# Save the model:
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, model_outpath)