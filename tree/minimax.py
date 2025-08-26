import math
import chess
import time

def evaluate_board(board: chess.Board):
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    value = 0
    for piece_type in piece_values:
        value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    value += 0.1 * board.legal_moves.count()
    
    return value

def minimax(board: chess.Board, depth, maximizing_player, positions_searched):
    positions_searched[0] += 1
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False, positions_searched)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True, positions_searched)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

def find_best_move_minimax(board: chess.Board, depth):
    start_time = time.time()
    positions_searched = [0]
    # For the bot (Black), we are minimizing
    best_move = None
    best_value = math.inf
    for move in board.legal_moves:
        board.push(move)
        # We look from the perspective of the next player (White), who is maximizing
        board_value = minimax(board, depth - 1, True, positions_searched)
        board.pop()
        if board_value < best_value:
            best_value = board_value
            best_move = move

    end_time = time.time()
    time_taken = end_time - start_time
    return best_move, time_taken, positions_searched[0]