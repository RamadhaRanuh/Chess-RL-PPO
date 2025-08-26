import chess
import math
import time
import random

# ----------------------------------------------------------------
# AI LOGIC (Optimized Monte Carlo Tree Search)
# ----------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900
}

def playout_policy(board):
    """
    A simple, fast heuristic to guide the MCTS simulation.
    Prioritizes good captures and avoids obviously bad moves.
    This makes simulations faster and more realistic than pure random moves.
    """
    legal_moves = list(board.legal_moves)
    
    # 1. Look for winning or equal captures (MVV-LVA style)
    good_captures = []
    for move in legal_moves:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                # If the captured piece is more or equally valuable, it's a good capture
                if PIECE_VALUES.get(victim.piece_type, 0) >= PIECE_VALUES.get(attacker.piece_type, 0):
                    good_captures.append(move)

    if good_captures:
        return random.choice(good_captures)

    # 2. If no good captures, play any non-losing move
    # (A simple check: don't move to a square attacked by a pawn)
    safe_moves = []
    for move in legal_moves:
        # Check if the destination square is attacked by an opponent's pawn
        if not board.is_attacked_by(not board.turn, move.to_square):
             safe_moves.append(move)
    
    if safe_moves:
        return random.choice(safe_moves)
        
    # 3. If all moves are into an attack, just play any random move
    return random.choice(legal_moves)


class MCTSNode:
    """A node in the Monte Carlo Tree."""
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(self.board.legal_moves)
    
    def uct_select_child(self):
        """Selects the best child node using the UCT formula."""
        C = 1.414 
        s = sorted(self.children, key=lambda c: c.wins / c.visits + C * math.sqrt(math.log(self.visits) / c.visits))
        return s[-1]

    def add_child(self, move, board):
        """Adds a new child node."""
        node = MCTSNode(board=board, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(node)
        return node

    def update(self, result):
        """Updates the node's statistics."""
        self.visits += 1
        self.wins += result

def mcts_search(root_board, num_simulations, positions_searched):
    """Performs the main MCTS search."""
    root_node = MCTSNode(board=root_board)

    for _ in range(num_simulations):
        node = root_node
        board = root_board.copy()
        positions_searched[0] += 1

        # 1. Selection
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
            board.push(node.move)

        # 2. Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            board.push(move)
            node = node.add_child(move, board)

        # 3. Simulation (with our smart playout policy)
        while not board.is_game_over():
            move = playout_policy(board) # <-- THE KEY OPTIMIZATION
            board.push(move)
        
        result = board.result()
        if result == '1-0': simulation_result = 1 if node.parent.board.turn == chess.WHITE else 0
        elif result == '0-1': simulation_result = 1 if node.parent.board.turn == chess.BLACK else 0
        else: simulation_result = 0.5

        # 4. Backpropagation
        while node is not None:
            node.update(simulation_result)
            simulation_result = 1 - simulation_result
            node = node.parent
    
    best_move = sorted(root_node.children, key=lambda c: c.visits)[-1].move
    return best_move

def find_best_move_mcts(board, time_limit_seconds=5):
    """The main entry point for the MCTS AI."""
    start_time = time.time()
    positions_searched = [0]
    num_simulations = 0

    # This version runs for a fixed number of simulations for consistency.
    # You should see a dramatic increase in simulations per second.
    target_simulations = 5000 # Let's aim higher now that it's faster

    best_move = mcts_search(board.copy(), target_simulations, positions_searched)
            
    end_time = time.time()
    
    if best_move is None and list(board.legal_moves):
        best_move = list(board.legal_moves)[0]
        
    return best_move, end_time - start_time, positions_searched[0]
