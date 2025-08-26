import chess
from chess import polyglot
import math
import time

# ----------------------------------------------------------------
# AI LOGIC (Alpha-Beta with Transposition Tables & Quiescence Search)
# ----------------------------------------------------------------

transposition_table = {}
EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

def evaluate_board(board):
    """
    Evaluates the board based on material count.
    """
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    value = 0
    for piece_type in PIECE_VALUES:
        value += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return value

def get_captured_piece_value(board, move):
    """
    Helper function to safely get the value of a captured piece,
    handling en passant captures correctly.
    """
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    
    # For regular captures, the captured piece is on the 'to_square'.
    # We need to look at the board *before* the move is made.
    captured_piece = board.piece_at(move.to_square)
    if captured_piece:
        return PIECE_VALUES[captured_piece.piece_type]
    return 0 # Should not happen for a capture move, but as a fallback.


def quiescence_search(board, alpha, beta, positions_searched):
    """
    Search only capture moves until a "quiet" position is reached.
    """
    positions_searched[0] += 1
    stand_pat_eval = evaluate_board(board)

    if stand_pat_eval >= beta:
        return beta
    alpha = max(alpha, stand_pat_eval)

    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    
    # --- FIX: Sort captures safely, handling en passant ---
    capture_moves.sort(key=lambda move: get_captured_piece_value(board, move), reverse=True)

    for move in capture_moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, positions_searched)
        board.pop()

        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def alphabeta_tt_q(board, depth, alpha, beta, maximizing_player, positions_searched):
    """
    The main search function combining all optimizations.
    """
    alpha_orig = alpha
    
    zobrist_hash = polyglot.zobrist_hash(board)
    if zobrist_hash in transposition_table:
        entry = transposition_table[zobrist_hash]
        if entry['depth'] >= depth:
            if entry['flag'] == EXACT: return entry['score']
            elif entry['flag'] == LOWER_BOUND: alpha = max(alpha, entry['score'])
            elif entry['flag'] == UPPER_BOUND: beta = min(beta, entry['score'])
            if alpha >= beta: return entry['score']

    if depth == 0:
        return quiescence_search(board, alpha, beta, positions_searched)

    positions_searched[0] += 1
    
    # Simple move ordering: check captures first.
    moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)

    if maximizing_player:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            eval = alphabeta_tt_q(board, depth - 1, alpha, beta, False, positions_searched)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        
        flag = EXACT
        if max_eval <= alpha_orig: flag = UPPER_BOUND
        elif max_eval >= beta: flag = LOWER_BOUND
        transposition_table[zobrist_hash] = {'score': max_eval, 'depth': depth, 'flag': flag}
        return max_eval
    else: # Minimizing player
        min_eval = math.inf
        for move in moves:
            board.push(move)
            eval = alphabeta_tt_q(board, depth - 1, alpha, beta, True, positions_searched)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        
        flag = EXACT
        if min_eval <= alpha_orig: flag = UPPER_BOUND
        elif min_eval >= beta: flag = LOWER_BOUND
        transposition_table[zobrist_hash] = {'score': min_eval, 'depth': depth, 'flag': flag}
        return min_eval

def find_best_move_alphabeta(board, depth):
    """
    The main entry point for the AI.
    """
    start_time = time.time()
    positions_searched = [0]
    best_move = None
    
    global transposition_table
    transposition_table = {}

    # The top-level call is slightly different as it needs to track the best move.
    if board.turn == chess.WHITE:
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            board_value = alphabeta_tt_q(board, depth - 1, -math.inf, math.inf, False, positions_searched)
            board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
    else: # Black (the bot)
        best_value = math.inf
        for move in board.legal_moves:
            board.push(move)
            board_value = alphabeta_tt_q(board, depth - 1, -math.inf, math.inf, True, positions_searched)
            board.pop()
            if board_value < best_value:
                best_value = board_value
                best_move = move
            
    end_time = time.time()
    
    if best_move is None and list(board.legal_moves):
        best_move = list(board.legal_moves)[0]
        
    return best_move, end_time - start_time, positions_searched[0]
