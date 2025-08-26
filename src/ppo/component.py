import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from torch.distributions import Categorical
from tqdm import tqdm

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


piece_to_plane = {
    ('P', chess.WHITE): 0,
    ('N', chess.WHITE): 1,
    ('B', chess.WHITE): 2,
    ('R', chess.WHITE): 3,
    ('Q', chess.WHITE): 4,
    ('K', chess.WHITE): 5,
    ('p', chess.BLACK): 6,
    ('n', chess.BLACK): 7,
    ('b', chess.BLACK): 8,
    ('r', chess.BLACK): 9,
    ('q', chess.BLACK): 10,
    ('k', chess.BLACK): 11,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess board object to a (17, 8, 8) PyTorch Tensor
    The 17 planes:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (p, n, b, r, q, k)
    - 12: Player to move (1 for white, 0 for black)
    - 13: White's kingside castling right
    - 14: White's queenside castling right
    - 15: Black's kingside castling right
    - 16: Black's queenside castling right
    """

    # Initialize a 17x8x8 NumPy array with zeros
    tensor_np = np.zeros((21, 8, 8), dtype=np.float32)

    # Populate piece planes (0 - 11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get the plane index for the piece type and color
            plane_idx = piece_to_plane[(piece.symbol(), piece.color)]

            # The board is read from A1 to H8, but we want a standard matrix view
            # rank (row) 7 -> 0, 6 -> 1, ..., 0 -> 7
            # file (col) 0 -> 0, 1 -> 1, ..., 7 -> 7
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor_np[plane_idx, 7-rank, file] = 1
    
    # Populate state planes (12 - 16)
    # Plane 12: PLayer to move
    if board.turn == chess.WHITE:
        tensor_np[12, :, :] = 1
    else:
        tensor_np[12, :, :] = 0 # Not necessary due to np.zeros, but for clarity

    # Plane 13 - 16: Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor_np[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor_np[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor_np[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor_np[16, :, :] = 1

    # Plane 17: En passant target
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor_np[17, 7 - rank, file] = 1

    # Plane 18: Has the position been repeated once?
    if board.is_repetition(2):
        tensor_np[18, :, :] = 1

    # Plane 19: Is the position about to be a 3-fold repetition draw?
    if board.is_repetition(3):
        tensor_np[19, :, :] = 1

    # Plane 20: Normalized total move count
    tensor_np[20, :, :] = min(1.0, board.fullmove_number / 100)

    return torch.from_numpy(tensor_np)


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for our chess bot.
    The network shares a convolutional body and has two heads:
    1. Policy Head (Actor): Outputs move probabilities.
    2. Value Head (Critic): Outputs a scalar value for the position
    """
    def __init__(self, num_actions=4672):
        super(ActorCritic, self).__init__()
        # --- Shared Convolutional Body ---
        # Input shape: (batch_size, 19, 8, 8) - updated for new tensor size
        self.conv1 = nn.Conv2d(21, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # The output of the conv layers will be (batch_size, 256, 8, 8)
        self.linear_input_size = 256 * 8 * 8

        # --- Critic Head ---
        self.value_head = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Outputs a value between -1 and 1
        )

        # --- Actor Head ---
        self.policy_head = nn.Sequential(
            nn.Linear(self.linear_input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_actions) # Outputs logits for each possible action
        )

    def forward(self, x):
        # Pass through the shared body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output for the linear layers
        x = x.view(-1, self.linear_input_size)

        # Calcualte value and policy
        value = self.value_head(x)
        policy_logits = self.policy_head(x)

        return policy_logits, value
    

def create_uci_move_map():
    """
    Creates a dictionary mapping all possible UCI moves to an integer index.
    Total actions: 4672
    """
    move_map = {}
    idx = 0

    # 1. Queen-like moves (rook and bishop moves)
    for from_sq in range(64):
        # Rook moves (up, down, left, right)
        for dr, df in [(1,0), (-1,0), (0, 1), (0, -1)]:
            for dist in range(1, 8):
                r, f = from_sq // 8, from_sq % 8
                nr, nf = r + dr * dist, f + df * dist
                if 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    move_map[chess.Move(from_sq, to_sq).uci()] = idx
                    idx += 1
                else:
                    # Pad to 7 moves for each direction
                    for _ in range(dist, 8):
                        move_map[f"dummy_{idx}"] = idx
                        idx += 1
                    break

        # Bishop moves (diagonals)
        for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for dist in range(1, 8):
                r, f = from_sq // 8, from_sq % 8
                nr, nf = r + dr * dist, f + df * dist
                if 0 <= nr < 8 and 0 <= nf < 8:
                    to_sq = nr * 8 + nf
                    move_map[chess.Move(from_sq, to_sq).uci()] = idx
                    idx += 1
                else:
                    for _ in range(dist, 8):
                        move_map[f"dummy_{idx}"] = idx
                        idx += 1
                    break

    # 2. Knight moves
    for from_sq in range(64):
        for dr, df in [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]:
            r, f = from_sq // 8, from_sq % 8
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                to_sq = nr * 8 + nf
                move_map[chess.Move(from_sq, to_sq).uci()] = idx
            else:
                move_map[f"dummy_{idx}"] = idx
            idx += 1

    # 3. Pawn promotions (underpromotions are handled here)
    for from_file in range(8):
        # White promotions (rank 6 to 7)
        from_sq = 6 * 8 + from_file
        for df in [-1, 0, 1]:
            to_file = from_file + df
            if 0 <= to_file < 8:
                to_sq = 7 * 8 + to_file
                for promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                     move_map[chess.Move(from_sq, to_sq, promotion).uci()] = idx
                     idx += 1
    
    # Black promotions (rank 1 to 0)
    for from_file in range(8):
        from_sq = 1 * 8 + from_file
        for df in [-1, 0, 1]:
            to_file = from_file + df
            if 0 <= to_file < 8:
                to_sq = 0 * 8 + to_file
                for promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                     move_map[chess.Move(from_sq, to_sq, promotion).uci()] = idx
                     idx += 1
    
    # We create the reverse mapping as well
    index_to_move_uci = {i: m for m, i in move_map.items() if not m.startswith("dummy")}
    
    return move_map, index_to_move_uci

MOVE_TO_INDEX_UCI, INDEX_TO_MOVE_UCI = create_uci_move_map()



def compute_advantages_and_returns(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    Computes advantages and returns for a trajectory using GAE.
    """
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    last_advantage = 0
    # The last value is the final reward
    last_return = rewards[-1]
    
    # Iterate backwards from the second to last step
    for t in reversed(range(len(rewards) - 1)):
        # The value of the next state is needed for the TD error
        next_value = values[t+1]
        
        # Calculate the TD error (delta)
        # delta = reward + gamma * V(s_t+1) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        
        # Calculate the advantage using the GAE formula
        # A(s_t) = delta + gamma * lambda * A(s_t+1)
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
        
        # Calculate the return for this step
        # R(t) = reward_t + gamma * R(t+1)
        returns[t] = rewards[t] + gamma * last_return
        last_return = returns[t]
        
    # The advantage for the final step is just its TD error
    advantages[-1] = rewards[-1] - values[-1]
    returns[-1] = rewards[-1]
    
    return advantages, returns

def compute_gae_and_returns(rewards, values, gamma=0.99, lambda_gae=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and returns.
    """
    # Detach values to prevent gradients from flowing back from the GAE calculation
    values = values.detach().cpu().numpy()
    rewards = rewards.cpu().numpy()
    
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    # We calculate next_value, which is V(s_{t+1})
    # For the last state, V(s_T) is 0 because the game is over.
    next_values = np.append(values[1:], 0)
    
    # GAE is calculated backwards from the last step to the first
    for t in reversed(range(len(rewards))):
        # TD-Error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] - values[t]
        
        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        last_advantage = delta + gamma * lambda_gae * last_advantage
        advantages[t] = last_advantage
        
    # Returns are calculated as advantages + values
    returns = advantages + values
    
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

def ppo_update(model, optimizer, trajectories, device, epochs=4, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01):
    """
    Performs the PPO update using GAE and correct reward assignment.
    """
    all_states = []
    all_actions = []
    all_log_probs = []
    all_advantages = []
    all_returns = []

    # 1. Process all trajectories to compute advantages and returns
    for trajectory, white_reward in trajectories: # The reward is from White's perspective
        
        # --- FIX 1: Correctly assign rewards based on player turn ---
        per_step_rewards = []
        for turn in trajectory["turns"]:
            # If it was White's move, the reward is white_reward (e.g., +1 for a win)
            # If it was Black's move, the reward is the opposite (e.g., -1 for a win by White)
            reward = white_reward if turn == chess.WHITE else -white_reward
            per_step_rewards.append(reward)
            
        # We need to change the final reward to be from the perspective of the LAST player
        # so the GAE calculation is correct for the final step.
        rewards_tensor = torch.zeros(len(trajectory["values"]))
        rewards_tensor[-1] = per_step_rewards[-1]

        game_values = torch.stack(trajectory["values"])
        
        # --- FIX 2: Use Generalized Advantage Estimation (GAE) ---
        advantages, returns = compute_gae_and_returns(rewards_tensor, game_values)
        
        # Normalize advantages across the batch for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store data for the batch
        all_states.extend(trajectory["states"])
        all_actions.extend(trajectory["actions"])
        all_log_probs.extend(trajectory["log_probs"])
        all_advantages.append(advantages)
        all_returns.append(returns)

    # Convert lists to tensors for batch processing
    states_tensor = torch.stack(all_states).to(device)
    actions_tensor = torch.stack(all_actions).to(device)
    old_log_probs_tensor = torch.stack(all_log_probs).detach().to(device)
    advantages_tensor = torch.cat(all_advantages).to(device)
    returns_tensor = torch.cat(all_returns).to(device)

    # 2. Perform PPO updates for a few epochs (your existing code is good here)
    model.train()
    total_policy_loss, total_value_loss, total_entropy = 0, 0, 0
    
    for _ in range(epochs):
        new_logits, new_values = model(states_tensor)
        dist = Categorical(F.softmax(new_logits, dim=-1))
        
        new_log_probs = dist.log_prob(actions_tensor.squeeze(-1))
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor.squeeze(-1))
        
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(new_values.squeeze(-1), returns_tensor)
        
        loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
    
    avg_metrics = {
        "policy_loss": total_policy_loss / epochs,
        "value_loss": total_value_loss / epochs,
        "entropy": total_entropy / epochs
    }
    return avg_metrics


class ChessDataset(Dataset):
    """
    Custom PyTorch Dataset for our chess data.
    """
    def __init__(self, csv_file, limit = None):
        """
        Args:
            csv_file (string): Path to the csv file with FENs and evaluations.
        """
        self.dataframe = pd.read_csv(csv_file)
        if limit:
            # Slice the dataframe to the specified limit
            self.dataframe = self.dataframe.head(limit)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the FEN string and evaluation from the dataframe
        fen = self.dataframe.iloc[idx]['FEN']
        evaluation = self.dataframe.iloc[idx]['Evaluation'].lstrip('#')


        # Create a board object
        board = chess.Board(fen)
        
        # Convert the board to our tensor representation
        board_tensor = board_to_tensor(board)
        

        # --- Normalize the evaluation score ---
        # The raw score is in centipawns. Let's clamp it to a reasonable range,
        # for example -1000 to +1000, which is like a +/- 10 pawn advantage.
        # Then, we can scale it to the [-1, 1] range.
        score = float(evaluation)
        score_clamped = max(min(score, 1000), -1000)
        # A simple scaling to [-1, 1]
        normalized_score = score_clamped / 1000.0
        
        # Convert score to a tensor
        eval_tensor = torch.tensor([normalized_score], dtype=torch.float32)

        return board_tensor, eval_tensor
    
def move_to_action(move: chess.Move) -> int:
    """
    Converts a chess.Move object to its corresponding action index.
    """
    uci = move.uci()

    if move.promotion == chess.QUEEN:
        uci = uci[:-1]
    try:
        return MOVE_TO_INDEX_UCI[uci]
    except KeyError:
        raise ValueError(f"Move {uci} not found in action mapping.")

def action_to_move(action_index: int, board: chess.Board) -> chess.Move | None:
    """
    Converts an action index back to a chess.Move object.
    It's crucial to check if the move is legal in the current board position.
    """
    uci = INDEX_TO_MOVE_UCI.get(action_index)
    if uci is None:
        return None
    try:
        return board.parse_uci(uci)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        # This can happen if the move is not pseudo-legal (e.g., pawn promotion from wrong rank)
        return None
    
def get_bot_move(board: chess.Board, model, device: torch.device) -> chess.Move:
    """
    Gets the best move for a given model in a given board position.
    """
    # 1. Convert board to tensor
    state_tensor = board_to_tensor(board).unsqueeze(0).to(device)

    with torch.no_grad():
        # 2. Get policy logits from the model
        policy_logits, _ = model(state_tensor)

    # 3. Mask illegal moves
    legal_move_indices = [move_to_action(move) for move in board.legal_moves]
    mask = torch.ones_like(policy_logits) * -1e9
    mask[0, legal_move_indices] = 0
    masked_logits = policy_logits + mask

    # 4. For evaluation, we deterministically choose the best move (argmax)
    #    instead of sampling from the probability distribution.
    best_action_index = torch.argmax(masked_logits).item()
    
    move = action_to_move(best_action_index)

    # Fallback in the rare case the chosen move is somehow illegal
    if move is None or move not in board.legal_moves:
        return np.random.choice(list(board.legal_moves))
        
    return move