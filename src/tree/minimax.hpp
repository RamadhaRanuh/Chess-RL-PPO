#pragma once

#include "chess.hpp"
#include "pst.hpp"
#include "zobrist.hpp"
#include <algorithm>
#include <vector>

struct Move {
    sf::Vector2i start;
    sf::Vector2i end;
    int score = 0; // For move ordering
};

const int AI_SEARCH_DEPTH = 4; // Increase the depth for more sophisticated bot

// --- Function Declarations ---
int evaluateBoard(const int board[8][8]);
int quiescenceSearch(int alpha, int beta, int board[8][8], sf::Vector2i wKing, sf::Vector2i bKing, bool wcK, bool wcQ, bool bcK, bool bcQ, sf::Vector2i epSquare, int currentPlayer, unsigned long long& calculations);
int minimax(int depth, int alpha, int beta, bool isMaximizingPlayer, int board[8][8], sf::Vector2i& wKing, sf::Vector2i& bKing, bool& wcK, bool& wcQ, bool& bcK, bool& bcQ, sf::Vector2i& epSquare, unsigned long long& calculations);
Move getAIBestMove(int board[8][8], int currentPlayer, sf::Vector2i wKing, sf::Vector2i bKing, bool wcK, bool wcQ, bool bcK, bool bcQ, sf::Vector2i epSquare, unsigned long long& calculations);
std::vector<Move> generateAllLegalMoves(int player, int board[8][8], sf::Vector2i wKing, sf::Vector2i bKing, sf::Vector2i epSquare, bool wcK, bool wcQ, bool bcK, bool bcQ);

void makeMove(Move move, int board[8][8], int& currentPlayer, GameState& gameState, int& winner, sf::Vector2i& whiteKingPos, sf::Vector2i& blackKingPos, bool& whiteCanCastleKingSide, bool& whiteCanCastleQueenSide, bool& blackCanCastleKingSide, bool& blackCanCastleQueenSide, sf::Vector2i& enPassantTargetSquare, map<string, int>& positionHistory, int& fiftyMoveCounter, vector<string>& moveHistory, sf::Vector2i& promotionSquare);


// --- Function Definitions ---

int evaluateBoard(const int board[8][8]) {
    int score = 0;
    map<int, int> pieceValues = {{1, 100}, {2, 320}, {3, 330}, {4, 500}, {5, 900}, {6, 20000}, {-1, -100}, {-2, -320}, {-3, -330}, {-4, -500}, {-5, -900}, {-6, -20000}};
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (board[r][c] != 0) {
                int piece = board[r][c];
                int piece_type = abs(piece);
                int sign = (piece > 0) ? 1 : -1;
                int square = r * 8 + c;
                int mirrored_square = (7 - r) * 8 + c;

                score += pieceValues.at(piece);
                if (sign == 1) { // White
                    if(piece_type == 1) score += PST::pawn_table[square];
                    if(piece_type == 2) score += PST::knight_table[square];
                    if(piece_type == 3) score += PST::bishop_table[square];
                    if(piece_type == 4) score += PST::rook_table[square];
                    if(piece_type == 5) score += PST::queen_table[square];
                } else { // Black
                    if(piece_type == 1) score -= PST::pawn_table[mirrored_square];
                    if(piece_type == 2) score -= PST::knight_table[mirrored_square];
                    if(piece_type == 3) score -= PST::bishop_table[mirrored_square];
                    if(piece_type == 4) score -= PST::rook_table[mirrored_square];
                    if(piece_type == 5) score -= PST::queen_table[mirrored_square];
                }
            }
        }
    }
    return score;
}

int quiescenceSearch(int alpha, int beta, int board[8][8], sf::Vector2i wKing, sf::Vector2i bKing, bool wcK, bool wcQ, bool bcK, bool bcQ, sf::Vector2i epSquare, int currentPlayer, unsigned long long& calculations) {
    calculations++;
    int stand_pat = evaluateBoard(board);

    if (currentPlayer == 1) { // Maximizing
        if (stand_pat >= beta) return beta;
        alpha = max(alpha, stand_pat);
    } else { // Minimizing
        if (stand_pat <= alpha) return alpha;
        beta = min(beta, stand_pat);
    }

    // Generate and score only capture moves
    vector<Move> captures = generateAllLegalMoves(currentPlayer, board, wKing, bKing, epSquare, wcK, wcQ, bcK, bcQ);
    captures.erase(remove_if(captures.begin(), captures.end(), [&](const Move& m) {
        return board[m.end.y][m.end.x] == 0;
    }), captures.end());
    sort(captures.begin(), captures.end(), [](const Move& a, const Move& b) { return a.score > b.score; });


    for (const auto& move : captures) {
        // --- Make Move ---
        int capturedPiece = board[move.end.y][move.end.x];
        board[move.end.y][move.end.x] = board[move.start.y][move.start.x];
        board[move.start.y][move.start.x] = 0;

        int score = quiescenceSearch(alpha, beta, board, wKing, bKing, wcK, wcQ, bcK, bcQ, {-1,-1}, -currentPlayer, calculations);
        
        // --- Unmake Move ---
        board[move.start.y][move.start.x] = board[move.end.y][move.end.x];
        board[move.end.y][move.end.x] = capturedPiece;

        if (currentPlayer == 1) {
            alpha = max(alpha, score);
            if (alpha >= beta) break;
        } else {
            beta = min(beta, score);
            if (beta <= alpha) break;
        }
    }
    return (currentPlayer == 1) ? alpha : beta;
}


int minimax(int depth, int alpha, int beta, bool isMaximizingPlayer, int board[8][8], sf::Vector2i& wKing, sf::Vector2i& bKing, bool& wcK, bool& wcQ, bool& bcK, bool& bcQ, sf::Vector2i& epSquare, unsigned long long& calculations) {
    calculations++;
    uint64_t hash = Zobrist::generate_hash(board, isMaximizingPlayer ? 1 : -1, wcK, wcQ, bcK, bcQ, epSquare);
    
    auto it = transposition_table.find(hash);
    if (it != transposition_table.end() && it->second.depth >= depth) {
        TTEntry entry = it->second;
        if (entry.flag == TTEntry::EXACT) return entry.score;
        if (entry.flag == TTEntry::LOWER_BOUND) alpha = max(alpha, entry.score);
        else if (entry.flag == TTEntry::UPPER_BOUND) beta = min(beta, entry.score);
        if (alpha >= beta) return entry.score;
    }

    if (depth == 0) return quiescenceSearch(alpha, beta, board, wKing, bKing, wcK, wcQ, bcK, bcQ, epSquare, isMaximizingPlayer ? 1 : -1, calculations);

    int player = isMaximizingPlayer ? 1 : -1;
    vector<Move> orderedMoves = generateAllLegalMoves(player, board, wKing, bKing, epSquare, wcK, wcQ, bcK, bcQ);

    if (orderedMoves.empty()) {
        sf::Vector2i kingPos = findKing(player, wKing, bKing);
        if (isSquareAttacked(kingPos.y, kingPos.x, -player, board, wKing, bKing, epSquare, wcK, wcQ, bcK, bcQ)) return isMaximizingPlayer ? -99999 - depth : 99999 + depth; // Checkmate, prefer faster mates
        return 0; // Stalemate
    }
    
    int bestValue = isMaximizingPlayer ? -100000 : 100000;
    auto bestFlag = TTEntry::UPPER_BOUND;

    for (const auto& move : orderedMoves) {
        // --- Make Move ---
        int piece = board[move.start.y][move.start.x];
        int capturedPiece = board[move.end.y][move.end.x];
        sf::Vector2i oldEpSquare = epSquare;
        bool oldWCK = wcK, oldWCQ = wcQ, oldBCK = bcK, oldBCQ = bcQ;
        sf::Vector2i oldKingPos = (player == 1) ? wKing : bKing;
        
        board[move.end.y][move.end.x] = piece;
        board[move.start.y][move.start.x] = 0;
        epSquare = {-1, -1}; // Reset en passant square

        if(abs(piece) == 6) { (player == 1) ? wKing = move.end : bKing = move.end; }
        if (piece == 6) { wcK = false; wcQ = false; }
        if (piece == -6) { bcK = false; bcQ = false; }
        if (piece == 4 && move.start.y == 7 && move.start.x == 7) wcK = false;
        if (piece == 4 && move.start.y == 7 && move.start.x == 0) wcQ = false;
        if (piece == -4 && move.start.y == 0 && move.start.x == 7) bcK = false;
        if (piece == -4 && move.start.y == 0 && move.start.x == 0) bcQ = false;
        if (abs(piece) == 1 && abs(move.start.y - move.end.y) == 2) epSquare = {move.start.x, (move.start.y + move.end.y) / 2};

        // --- Recurse ---
        int value = minimax(depth - 1, alpha, beta, !isMaximizingPlayer, board, wKing, bKing, wcK, wcQ, bcK, bcQ, epSquare, calculations);
        
        // --- Unmake Move ---
        board[move.start.y][move.start.x] = piece;
        board[move.end.y][move.end.x] = capturedPiece;
        epSquare = oldEpSquare;
        wcK = oldWCK; wcQ = oldWCQ; bcK = oldBCK; bcQ = oldBCQ;
        if (player == 1) wKing = oldKingPos; else bKing = oldKingPos;
        
        // --- Update Alpha/Beta ---
        if (isMaximizingPlayer) {
            if (value > bestValue) {
                bestValue = value;
                bestFlag = TTEntry::EXACT;
            }
            alpha = max(alpha, bestValue);
        } else {
            if (value < bestValue) {
                bestValue = value;
                bestFlag = TTEntry::EXACT;
            }
            beta = min(beta, bestValue);
        }
        if (alpha >= beta) {
            bestFlag = TTEntry::LOWER_BOUND;
            break;
        }
    }
    
    transposition_table[hash] = {bestValue, depth, bestFlag};
    return bestValue;
}

std::vector<Move> generateAllLegalMoves(int player, int board[8][8], sf::Vector2i wKing, sf::Vector2i bKing, sf::Vector2i epSquare, bool wcK, bool wcQ, bool bcK, bool bcQ) {
    vector<Move> allMoves;
    map<int, int> pieceValues = {{1, 100}, {2, 320}, {3, 330}, {4, 500}, {5, 900}, {6, 20000}};

    for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) {
        if (board[r][c] * player > 0) {
            vector<sf::Vector2i> moves = getLegalMoves(board[r][c], r, c, board, wKing, bKing, epSquare, wcK, wcQ, bcK, bcQ);
            for (const auto& move : moves) {
                int moveScore = 0;
                int victim = abs(board[move.y][move.x]);
                if (victim != 0) { // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
                    moveScore = pieceValues[victim] * 10 - pieceValues[abs(board[r][c])];
                }
                allMoves.push_back({{c, r}, move, moveScore});
            }
        }
    }
    // Sort moves to improve alpha-beta pruning effectiveness
    sort(allMoves.begin(), allMoves.end(), [](const Move& a, const Move& b) { return a.score > b.score; });
    return allMoves;
}

Move getAIBestMove(int board[8][8], int currentPlayer, sf::Vector2i wKing, sf::Vector2i bKing, bool wcK, bool wcQ, bool bcK, bool bcQ, sf::Vector2i epSquare, unsigned long long& calculations) {
    calculations = 0;
    transposition_table.clear();
    Move bestMove;
    bool isMaximizing = (currentPlayer == 1);
    int bestValue = isMaximizing ? -100001 : 100001;
    int alpha = -100000, beta = 100000;

    vector<Move> orderedMoves = generateAllLegalMoves(currentPlayer, board, wKing, bKing, epSquare, wcK, wcQ, bcK, bcQ);

    for (const auto& move : orderedMoves) {
        // --- Make Move ---
        int piece = board[move.start.y][move.start.x];
        int capturedPiece = board[move.end.y][move.end.x];
        sf::Vector2i oldEpSquare = epSquare;
        bool oldWCK = wcK, oldWCQ = wcQ, oldBCK = bcK, oldBCQ = bcQ;
        sf::Vector2i oldWKing = wKing, oldBKing = bKing;
        
        board[move.end.y][move.end.x] = piece;
        board[move.start.y][move.start.x] = 0;
        epSquare = {-1, -1};

        if(abs(piece) == 6) { (currentPlayer == 1) ? wKing = move.end : bKing = move.end; }
        if (piece == 6) { wcK = false; wcQ = false; }
        if (piece == -6) { bcK = false; bcQ = false; }
        if (piece == 4 && move.start.y == 7 && move.start.x == 7) wcK = false;
        if (piece == 4 && move.start.y == 7 && move.start.x == 0) wcQ = false;
        if (piece == -4 && move.start.y == 0 && move.start.x == 7) bcK = false;
        if (piece == -4 && move.start.y == 0 && move.start.x == 0) bcQ = false;
        if (abs(piece) == 1 && abs(move.start.y - move.end.y) == 2) epSquare = {move.start.x, (move.start.y + move.end.y) / 2};
        
        int boardValue = minimax(AI_SEARCH_DEPTH - 1, alpha, beta, !isMaximizing, board, wKing, bKing, wcK, wcQ, bcK, bcQ, epSquare, calculations);
        
        // --- Unmake Move ---
        board[move.start.y][move.start.x] = piece;
        board[move.end.y][move.end.x] = capturedPiece;
        epSquare = oldEpSquare;
        wcK = oldWCK; wcQ = oldWCQ; bcK = oldBCK; bcQ = oldBCQ;
        wKing = oldWKing; bKing = oldBKing;

        if (isMaximizing) {
            if (boardValue > bestValue) {
                bestValue = boardValue;
                bestMove = move;
            }
            alpha = max(alpha, bestValue);
        } else {
            if (boardValue < bestValue) {
                bestValue = boardValue;
                bestMove = move;
            }
            beta = min(beta, bestValue);
        }
    }
    return bestMove;
}


void makeMove(Move move, int board[8][8], int& currentPlayer, GameState& gameState, int& winner, sf::Vector2i& whiteKingPos, sf::Vector2i& blackKingPos, bool& whiteCanCastleKingSide, bool& whiteCanCastleQueenSide, bool& blackCanCastleKingSide, bool& blackCanCastleQueenSide, sf::Vector2i& enPassantTargetSquare, map<string, int>& positionHistory, int& fiftyMoveCounter, vector<string>& moveHistory, sf::Vector2i& promotionSquare) {
    sf::Vector2i oldPos = move.start;
    sf::Vector2i newPos = move.end;
    int selectedPiece = board[oldPos.y][oldPos.x];
    bool isCapture = (board[newPos.y][newPos.x] != 0) || (abs(selectedPiece) == 1 && newPos == enPassantTargetSquare);
    bool isPawnMove = (abs(selectedPiece) == 1);
    if (isPawnMove || isCapture) fiftyMoveCounter = 0; else fiftyMoveCounter++;
    string san = moveToSAN(selectedPiece, newPos, oldPos, board, isCapture, false, false, false, 0);
    moveHistory.push_back(san);
    if (abs(selectedPiece) == 6 && abs(newPos.x - oldPos.x) == 2) {
        int rookCol = (newPos.x > oldPos.x) ? 7 : 0;
        int newRookCol = (newPos.x > oldPos.x) ? 5 : 3;
        board[oldPos.y][newRookCol] = board[oldPos.y][rookCol]; board[oldPos.y][rookCol] = 0;
    }
    if (abs(selectedPiece) == 1 && newPos == enPassantTargetSquare) board[oldPos.y][newPos.x] = 0;
    if (selectedPiece == 6) { whiteCanCastleKingSide = false; whiteCanCastleQueenSide = false; }
    else if (selectedPiece == -6) { blackCanCastleKingSide = false; blackCanCastleQueenSide = false; }
    else if (selectedPiece == 4) { if(oldPos.x == 7 && oldPos.y == 7) whiteCanCastleKingSide = false; if(oldPos.x == 0 && oldPos.y == 7) whiteCanCastleQueenSide = false; }
    else if (selectedPiece == -4) { if(oldPos.x == 7 && oldPos.y == 0) blackCanCastleKingSide = false; if(oldPos.x == 0 && oldPos.y == 0) blackCanCastleQueenSide = false; }
    board[newPos.y][newPos.x] = selectedPiece;
    board[oldPos.y][oldPos.x] = 0;
    if(abs(selectedPiece) == 6) (selectedPiece > 0) ? whiteKingPos = newPos : blackKingPos = newPos;
    bool isPromotion = (abs(selectedPiece) == 1) && ((selectedPiece > 0 && newPos.y == 0) || (selectedPiece < 0 && newPos.y == 7));
    if (isPromotion){
        gameState = PROMOTING; promotionSquare = newPos;
    } else {
        if (abs(selectedPiece) == 1 && abs(oldPos.y - newPos.y) == 2) enPassantTargetSquare = {newPos.x, (oldPos.y + newPos.y) / 2};
        else enPassantTargetSquare = {-1, -1};
        updateGameStatus(currentPlayer, gameState, winner, board, whiteKingPos, blackKingPos, enPassantTargetSquare, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, positionHistory, fiftyMoveCounter);
    }
}