import chess
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from torch.distributions import Categorical
from tqdm import tqdm
from component import board_to_tensor, move_to_action, action_to_move, create_uci_move_map, ActorCritic,  ppo_update

MOVE_TO_INDEX_UCI, INDEX_TO_MOVE_UCI = create_uci_move_map()
NUM_ACTIONS = 4672 


def run_self_play_game(model, device, max_moves = 250):
    """
    Simulates one full game of self-play.
    Returns the collected trajectory data needed for PPO.
    """
    # Lists to store the data for the entire game
    trajectory = {"states": [], "actions": [], "log_probs": [], "values": [], "turns": []}
    board = chess.Board()
    model.eval() 

    with torch.no_grad(): # We are not training here, just collecting data
        while not board.is_game_over() and board.fullmove_number < max_moves:
            state_tensor = board_to_tensor(board).unsqueeze(0).to(device)
            policy_logits, value = model(state_tensor)
                        
            legal_move_indices = [move_to_action(move) for move in board.legal_moves]
            mask = torch.ones_like(policy_logits) * -1e9 
            mask[0, legal_move_indices] = 0
            
            masked_logits = policy_logits + mask
            probs = F.softmax(masked_logits, dim=-1)
            
            dist = Categorical(probs)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
            
            trajectory["turns"].append(board.turn)
            trajectory["states"].append(state_tensor.squeeze(0))
            trajectory["actions"].append(action_index)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value.squeeze())
            
            move = action_to_move(action_index.item(), board)

            if move is None or move not in board.legal_moves:
                move = np.random.choice(list(board.legal_moves))
            board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0": # White wins
        white_reward = 1.0
    elif result == "0-1": # Black wins
        white_reward = -1.0
    else: # Draw
        white_reward = 0.0
        

    return trajectory, white_reward, [board.fen()]


def self_play_worker(model_weights, device_str):
    """
    A worker function that plays one game and returns the result.
    It reconstructs the model inside the process.
    """
    # Each process needs to have its own model instance
    device = torch.device(device_str)
    model = ActorCritic(num_actions=NUM_ACTIONS).to(device)
    model.load_state_dict(model_weights)
    
    trajectory, reward, _ = run_self_play_game(model, device)

    cpu_trajectory = {}
    for key, value_list in trajectory.items():
        # Check if the list contains tensors before trying to move them
        if value_list and isinstance(value_list[0], torch.Tensor):
            cpu_trajectory[key] = [t.cpu() for t in value_list]
        else:
            # If not a list of tensors (e.g., 'turns'), just copy it
            cpu_trajectory[key] = value_list
            
    return cpu_trajectory, reward

if __name__ == "__main__":
    # --- Setup for Training ---
    mp.set_start_method('spawn', force=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Instantiate the main model
    model = ActorCritic(num_actions=NUM_ACTIONS).to(device)

    # --- Load Weights to Resume Training ---
    try:
        model.load_state_dict(torch.load('ppo_trained_bot.pth', weights_only=True))
        print("Resuming training from 'ppo_trained_bot.pth'.")
    except FileNotFoundError:
        try:
            model.load_state_dict(torch.load('critic_pretrained_weights.pth', weights_only=True), strict=False)
            print("Starting training from 'critic_pretrained_weights.pth'.")
        except FileNotFoundError:
            print("No saved weights found. Starting training from scratch.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
    games_per_update = 50 # You can increase this now, e.g., to 50 or 100

    # --- Main Training Loop with Parallelization ---
    print("\n--- Starting Parallel PPO Training ---")
    for update_step in range(10): # Run for many steps
        print(f"--- Starting Update Step {update_step + 1} ---")
        
        # Get the current model weights to send to workers
        model_weights = model.state_dict()
        
        # Use a Pool to run games in parallel
        # mp.cpu_count() uses all available CPU cores
        total_cores = mp.cpu_count()
        num_workers = max(1, total_cores - 24)
        print(f"  - Collecting {games_per_update} games using {num_workers} parallel workers...")
        with mp.Pool(processes=num_workers) as pool:
            worker_args = [(model_weights, device_str) for _ in range(games_per_update)]
            batch_trajectories = list(tqdm(pool.starmap(self_play_worker, worker_args), total=games_per_update))
        
        batch_rewards = [r for t, r in batch_trajectories]
        
        print("  - All games finished. Performing PPO update...")
        metrics = ppo_update(model, optimizer, batch_trajectories, device)
        avg_reward = sum(batch_rewards) / len(batch_rewards)

        scheduler.step()
        
        print(
            f"Update: {update_step + 1}, "
            f"Avg Reward: {avg_reward:.4f}, "
            f"Value Loss: {metrics['value_loss']:.4f}, "
            f"Entropy: {metrics['entropy']:.4f}"
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Checkpointing
        torch.save(model.state_dict(), 'ppo_trained_bot.pth')

    print("--- Training Finished ---")