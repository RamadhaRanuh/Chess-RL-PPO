import chess
import numpy as np
import torch
from tqdm import tqdm
from component import board_to_tensor, create_uci_move_map, ActorCritic

# --- Global Setup ---
NUM_ACTIONS = 4672
MOVE_TO_INDEX_UCI, INDEX_TO_MOVE_UCI = create_uci_move_map()

# --- Helper Functions ---
def move_to_action(move: chess.Move) -> int:
    uci = move.uci()
    if move.promotion == chess.QUEEN:
        uci = uci[:-1]
    try:
        return MOVE_TO_INDEX_UCI[uci]
    except KeyError:
        raise ValueError(f"Move {uci} not found in action mapping.")

def action_to_move(action_index: int) -> chess.Move:
    uci = INDEX_TO_MOVE_UCI.get(action_index)
    if uci:
        return chess.Move.from_uci(uci)
    return None

def get_bot_move(board: chess.Board, model, device: torch.device) -> chess.Move:
    state_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
    
    legal_move_indices = [move_to_action(move) for move in board.legal_moves]
    mask = torch.ones_like(policy_logits) * -1e9
    mask[0, legal_move_indices] = 0
    masked_logits = policy_logits + mask
    
    best_action_index = torch.argmax(masked_logits).item()
    move = action_to_move(best_action_index)
    
    if move is None or move not in board.legal_moves:
        return np.random.choice(list(board.legal_moves))
    return move

# --- Match Logic ---
def run_match(model_v2, model_v1, device, num_games=50): # Increased games for better stats
    scores = {"v2_wins": 0, "v1_wins": 0, "draws": 0}
    
    for i in tqdm(range(num_games), desc="Playing Games"):
        board = chess.Board()
        if i % 2 == 0:
            players = {chess.WHITE: model_v2, chess.BLACK: model_v1}
            player_names = {chess.WHITE: "v2", chess.BLACK: "v1"}
        else:
            players = {chess.WHITE: model_v1, chess.BLACK: model_v2}
            player_names = {chess.WHITE: "v1", chess.BLACK: "v2"}
            
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 150:
            current_player_model = players[board.turn]
            move = get_bot_move(board, current_player_model, device)
            board.push(move)
            
        result = board.result(claim_draw=True)
        
        if result == "1-0":
            scores[f"{player_names[chess.WHITE]}_wins"] += 1
        elif result == "0-1":
            scores[f"{player_names[chess.BLACK]}_wins"] += 1
        else:
            scores["draws"] += 1
            
    print("\n--- Match Finished ---")
    print(f"Final Scores after {num_games} games:")
    print(f"  - Fully Trained Bot (v2) Wins: {scores['v2_wins']}")
    print(f"  - Critic-Only Bot (v1) Wins: {scores['v1_wins']}")
    print(f"  - Draws: {scores['draws']}")
    
    win_rate_v2 = (scores['v2_wins'] + 0.5 * scores['draws']) / num_games
    if win_rate_v2 > 0.01 and win_rate_v2 < 0.99: # Avoid extreme values
        elo_diff = -400 * np.log10(1 / win_rate_v2 - 1)
        print(f"\nApproximate Elo difference: v2 is ~{elo_diff:.0f} points stronger than v1.")

if __name__ == "__main__":
    device = torch.device("cpu")

    # --- Load Model V2 (PPO-trained) ---
    print("Loading fully trained PPO model (v2)...")
    model_v2 = ActorCritic(num_actions=NUM_ACTIONS).to(device)
    model_v2.load_state_dict(torch.load('ppo_trained_bot.pth', map_location=device))
    model_v2.eval()
    print("Model v2 loaded successfully.")

    # --- Load Model V1 (Critic-only) ---
    print("\nLoading original critic-only model (v1)...")
    model_v1 = ActorCritic(num_actions=NUM_ACTIONS).to(device)
    try:
        pretrained_dict = torch.load('critic_pretrained_weights.pth', map_location=device)
        model_dict = model_v1.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model_v1.load_state_dict(model_dict)
        print("Model v1 loaded successfully.")
    except FileNotFoundError:
        print("Could not find critic_pretrained_weights.pth. Model v1 will be random.")
    model_v1.eval()

    # --- Run the match ---
    run_match(model_v2, model_v1, device)