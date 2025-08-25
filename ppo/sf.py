import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from component import piece_to_plane, board_to_tensor, ChessDataset, ActorCritic
from torch.utils.data import random_split
from torch.amp import GradScaler, autocast
import os

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    weights_path = 'critic_pretrained_weights.pth'

    for param in model.policy_head.parameters():
                param.requires_grad = False

    learning_rate = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()
    

    if os.path.exists(weights_path):
        print(f"Found pre-trained weights at '{weights_path}'.")
        # Load the existing weights into the model.
        # We use strict=False to be safe, in case the model architectures differ slightly.
        model.load_state_dict(torch.load(weights_path), strict=False)
        print("Weights loaded successfully. Pre-train this instance")

    else:
        print(f"No pre-trained weights found. Starting pre-training process...")       
        
    try:
        # Let's use the main dataset file
        csv_path = '../dataset/ChessData.csv'
        chess_dataset = ChessDataset(csv_file=csv_path, limit = 6000000)
        print(f"Successfully loaded {len(chess_dataset)} positions from {csv_path}")

    except FileNotFoundError:
        print("Error: Make sure 'ChessData.csv' is in the same directory as the script.")
        print("You can download it from the Kaggle link you provided.")
        exit() # Exit if the data isn't found


    dataset_size = len(chess_dataset)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(chess_dataset, [train_size, val_size])

    batch_size = 1024
    num_cores = 8 # Example value
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores, pin_memory=True)

    num_epochs = 20 # A few epochs are enough for a demonstration
    # (For real training, you'd run this for many more epochs)

    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    print("\n--- Starting Critic Pre-training ---")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        
        for i, (board_tensors, true_evals) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            # Move tensors to the correct device
            board_tensors = board_tensors.to(device)
            true_evals = true_evals.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, predicted_evals = model(board_tensors)
                loss = criterion(predicted_evals, true_evals)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)


        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for board_tensors, true_evals in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                board_tensors = board_tensors.to(device)
                true_evals = true_evals.to(device)
                
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    _, predicted_evals = model(board_tensors)
                    batch_loss = criterion(predicted_evals, true_evals)

                total_val_loss += batch_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        

        print(f"\n--- Epoch {epoch+1} Finished ---")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}\n")

    print("--- Pre-training Finished ---")

    torch.save(model.state_dict(), 'critic_pretrained_weights.pth')


