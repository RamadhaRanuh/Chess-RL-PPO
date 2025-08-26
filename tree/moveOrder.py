import chess
from chess import polyglot
import math
import time

# ----------------------------------------------------------------
# OPTIMIZED AI LOGIC
# ----------------------------------------------------------------

# --- Heuristic Tables & Constants ---
MAX_PLY = 32
killer_moves = [[None, None] for _ in range(MAX_PLY)]  # Store 2 killers per ply
history_scores = [[[0] * 64 for _ in range(64)] for _ in range(2)]
transposition_table = {}
EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2

# Reduced precision piece values for faster computation
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# Compressed PSTs - using tuples for faster lookup
PAWN_PST = (
    0,  0,  0,  0,  0,  0,  0,  0, 50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10, 5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0, 5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5, 0,  0,  0,  0,  0,  0,  0,  0
)
KNIGHT_PST = (
    -50,-40,-30,-30,-30,-30,-40,-50, -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30, -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30, -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50
)
BISHOP_PST = (
    -20,-10,-10,-10,-10,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10, -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10, -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10, -20,-10,-10,-10,-10,-10,-10,-20
)
ROOK_PST = (
    0,  0,  0,  5,  5,  0,  0,  0, -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5,
    5, 10, 10, 10, 10, 10, 10,  5, 0,  0,  0,  0,  0,  0,  0,  0
)
QUEEN_PST = (
    -20,-10,-10, -5, -5,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10, -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5, -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10, -20,-10,-10, -5, -5,-10,-10,-20
)
KING_PST = (
    20, 30, 10,  0,  0, 10, 30, 20, 20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10, -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30
)

PIECE_PSTS = {
    chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST, chess.KING: KING_PST
}

def evaluate_board(board):
    """Optimized board evaluation with early termination checks."""
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    # Fast material counting using bit operations where possible
    for piece_type in range(1, 7):  # PAWN to KING
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)
        piece_value = PIECE_VALUES[piece_type]
        score += len(white_pieces) * piece_value - len(black_pieces) * piece_value
        
        # PST evaluation
        if piece_type in PIECE_PSTS:
            pst = PIECE_PSTS[piece_type]
            for square in white_pieces:
                score += pst[square]
            for square in black_pieces:
                score -= pst[square ^ 56]  # Flip for black
    
    return score

def get_capture_score(board, move):
    """Optimized MVV-LVA scoring."""
    if board.is_en_passant(move):
        return 1100  # Pawn takes pawn + bonus
    
    victim = board.piece_at(move.to_square)
    if not victim:
        return 0
    
    attacker = board.piece_at(move.from_square)
    if not attacker:
        return 0
    
    # MVV-LVA: prioritize high-value victims and low-value attackers
    return PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type] + 1000

def quiescence_search(board, alpha, beta, positions_searched):
    """Enhanced quiescence search with better pruning."""
    positions_searched[0] += 1
    
    # Check for immediate tactical threats
    if board.is_check():
        # If in check, must search all legal moves, not just captures
        moves = list(board.legal_moves)
    else:
        # Stand pat evaluation
        stand_pat_eval = evaluate_board(board)
        if stand_pat_eval >= beta:
            return beta
        alpha = max(alpha, stand_pat_eval)
        
        # Delta pruning - if even capturing the most valuable piece can't improve alpha
        if stand_pat_eval + 900 < alpha:  # Queen value as max gain
            return alpha
        
        # Only search captures and promotions
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    
    # Sort captures by MVV-LVA
    moves.sort(key=lambda m: get_capture_score(board, m), reverse=True)
    
    for move in moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, positions_searched)
        board.pop()
        
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    
    return alpha

def score_move(board, move, ply, tt_best_move=None):
    """Enhanced move ordering with better heuristics."""
    # Hash move gets highest priority
    if tt_best_move and move == tt_best_move:
        return 100000
    
    # Promotions
    if move.promotion:
        return 20000 + PIECE_VALUES[move.promotion]
    
    # Captures sorted by MVV-LVA
    if board.is_capture(move):
        return get_capture_score(board, move)
    
    # Killer moves (non-captures that caused beta cutoffs)
    if ply < MAX_PLY:
        for killer in killer_moves[ply]:
            if killer == move:
                return 900
    
    # History heuristic
    return history_scores[board.turn][move.from_square][move.to_square]

def alphabeta_ordered(board, depth, ply, alpha, beta, maximizing_player, positions_searched):
    """Enhanced alpha-beta with better transposition table usage."""
    alpha_orig = alpha
    tt_best_move = None
    
    # Transposition table lookup
    zobrist_hash = polyglot.zobrist_hash(board)
    if zobrist_hash in transposition_table:
        entry = transposition_table[zobrist_hash]
        tt_best_move = entry.get('best_move')
        
        # Use stored result if depth is sufficient
        if entry['depth'] >= depth:
            if entry['flag'] == EXACT:
                return entry['score']
            elif entry['flag'] == LOWER_BOUND:
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == UPPER_BOUND:
                beta = min(beta, entry['score'])
            
            if alpha >= beta:
                return entry['score']
    
    # Leaf node - go to quiescence search
    if depth == 0:
        return quiescence_search(board, alpha, beta, positions_searched)
    
    positions_searched[0] += 1
    
    # Generate and sort moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -99999 + ply if maximizing_player else 99999 - ply
        else:
            return 0  # Stalemate
    
    # Sort moves for better alpha-beta pruning
    legal_moves.sort(key=lambda m: score_move(board, m, ply, tt_best_move), reverse=True)
    
    best_move_this_node = None
    
    if maximizing_player:
        max_eval = -math.inf
        for i, move in enumerate(legal_moves):
            board.push(move)
            
            # Late Move Reduction (LMR) - search later moves with reduced depth
            if i >= 4 and depth >= 3 and not board.is_capture(move) and not move.promotion and not board.is_check():
                eval_score = alphabeta_ordered(board, depth - 2, ply + 1, alpha, beta, False, positions_searched)
                if eval_score <= alpha:  # If LMR fails low, skip full search
                    board.pop()
                    continue
            
            # Full depth search
            eval_score = alphabeta_ordered(board, depth - 1, ply + 1, alpha, beta, False, positions_searched)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move_this_node = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # Beta cutoff - update killers and history
                if not board.is_capture(move) and ply < MAX_PLY:
                    # Update killer moves
                    if killer_moves[ply][0] != move:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = move
                    # Update history
                    history_scores[board.turn][move.from_square][move.to_square] += depth * depth
                break
        
        result = max_eval
    else:
        min_eval = math.inf
        for i, move in enumerate(legal_moves):
            board.push(move)
            
            # Late Move Reduction
            if i >= 4 and depth >= 3 and not board.is_capture(move) and not move.promotion and not board.is_check():
                eval_score = alphabeta_ordered(board, depth - 2, ply + 1, alpha, beta, True, positions_searched)
                if eval_score >= beta:
                    board.pop()
                    continue
            
            eval_score = alphabeta_ordered(board, depth - 1, ply + 1, alpha, beta, True, positions_searched)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move_this_node = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                if not board.is_capture(move) and ply < MAX_PLY:
                    if killer_moves[ply][0] != move:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = move
                    history_scores[board.turn][move.from_square][move.to_square] += depth * depth
                break
        
        result = min_eval
    
    # Store in transposition table
    flag = EXACT
    if result <= alpha_orig:
        flag = UPPER_BOUND
    elif result >= beta:
        flag = LOWER_BOUND
    
    transposition_table[zobrist_hash] = {
        'score': result, 
        'depth': depth, 
        'flag': flag, 
        'best_move': best_move_this_node
    }
    
    return result

def find_best_move_ordered(board, max_depth, time_limit=None):
    """Enhanced iterative deepening with time management."""
    start_time = time.time()
    positions_searched = [0]
    best_move = None
    
    # Clear search tables
    global transposition_table, killer_moves, history_scores
    transposition_table.clear()
    killer_moves = [[None, None] for _ in range(MAX_PLY)]
    # Don't clear history - it helps across moves
    
    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        iter_start = time.time()
        best_move_this_iter = None
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Sort moves by previous iteration's results (stored in TT)
        legal_moves.sort(key=lambda m: score_move(board, m, 0), reverse=True)
        
        if board.turn == chess.WHITE:
            best_value = -math.inf
            for move in legal_moves:
                board.push(move)
                board_value = alphabeta_ordered(board, depth - 1, 1, -math.inf, math.inf, False, positions_searched)
                board.pop()
                
                if board_value > best_value:
                    best_value = board_value
                    best_move_this_iter = move
                
                # Time check
                if time_limit and (time.time() - start_time) > time_limit:
                    return best_move or best_move_this_iter, time.time() - start_time, positions_searched[0]
        else:
            best_value = math.inf
            for move in legal_moves:
                board.push(move)
                board_value = alphabeta_ordered(board, depth - 1, 1, -math.inf, math.inf, True, positions_searched)
                board.pop()
                
                if board_value < best_value:
                    best_value = board_value
                    best_move_this_iter = move
                
                # Time check
                if time_limit and (time.time() - start_time) > time_limit:
                    return best_move or best_move_this_iter, time.time() - start_time, positions_searched[0]
        
        best_move = best_move_this_iter
        
        # Early termination for forced wins/losses
        if abs(best_value) > 90000:
            break
        
        # Time management - if this iteration took too long, don't start the next
        iter_time = time.time() - iter_start
        if time_limit and (time.time() - start_time + iter_time * 3) > time_limit:
            break
    
    return best_move or legal_moves[0], time.time() - start_time, positions_searched[0]