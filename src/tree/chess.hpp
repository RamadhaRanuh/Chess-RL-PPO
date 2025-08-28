#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
using namespace std;

enum GameState { CHOOSING_COLOR, PLAYING, PROMOTING, GAME_OVER };

// Helper to load a texture and handle errors
bool loadPieceTexture(map<int, sf::Texture>& textures, int piece, const string& filename){
    if(!textures[piece].loadFromFile(filename)){
        cerr << "Failed to load" << filename << endl;
        return false;
    }
    return true;
}

// Helper function to load the board state from a FEN string
void loadBoardFromFen(int board[8][8], const string& fen, sf::Vector2i& wKing, sf::Vector2i& bKing, bool& wcK, bool& wcQ, bool& bcK, bool& bcQ){
    for(int i = 0; i < 8; i++) for(int j = 0; j < 8; j++) board[i][j] = 0;

    map<char, int> pieceMap = {
        {'P', 1}, {'N', 2}, {'B', 3}, {'R', 4}, {'Q', 5}, {'K', 6},
        {'p', -1}, {'n', -2}, {'b', -3}, {'r', -4}, {'q', -5}, {'k', -6}
    };

    int row = 0, col = 0;
    size_t i = 0;
    for (; i < fen.length(); i++){
        char c = fen[i];
        if (c == ' '){
            i++;
            break;
        }

        if (c == '/'){
            row++;
            col = 0;
        } else if (isdigit(c)){
            col += c - '0';
        } else {
            if (row < 8 && col < 8) {
                if (pieceMap[c] == 6) wKing = {col, row};
                if (pieceMap[c] == -6) bKing = {col, row};

                board[row][col] = pieceMap[c];
                col++;
            }
        }
    }

    // Skip active color
    while(i < fen.length() && fen[i] != ' ') i++;
    i++;

    // Parse castling rights
    string castlingRights;
    while(i < fen.length() && fen[i] != ' '){
        castlingRights += fen[i];
        i++;
    }
    wcK = (castlingRights.find('K') != std::string::npos);
    wcQ = (castlingRights.find('Q') != std::string::npos);
    bcK = (castlingRights.find('k') != std::string::npos);
    bcQ = (castlingRights.find('q') != std::string::npos);
}

string generateStateString(const int board[8][8], int currentPlayer, bool wcK, bool wcQ, bool bcK, bool bcQ, const sf::Vector2i& enPassant){
    string state = "";
    for (int r = 0; r < 8; r++){
        for(int c = 0; c < 8; c++){
            state += to_string(board[r][c]) + ",";
        }
    }
    state += (currentPlayer == 1 ? "w" : "b");
    if (wcK) state += "K";
    if (wcQ) state += "Q";
    if (bcK) state += "k";
    if (bcQ) state += "q";
    state += to_string(enPassant.x) + "," + to_string(enPassant.y);
    return state;
}

// Helper function to convert board coordinates to algebraic notation
string squareToAlgebraic(const sf::Vector2i& pos){
    string s = "";
    s += (char)('a' + pos.x);
    s += (char)('8' - pos.y);
    return s;
}

// Finds the king of a given color on the board.
sf::Vector2i findKing(int color, const sf::Vector2i& wKing, const sf::Vector2i& bKing) {
    return (color == 1) ? wKing : bKing;
}

vector<sf::Vector2i> getPseudoLegalMoves(int piece, int row, int col, const int board[8][8], const sf::Vector2i& enPassantTargetSquare, bool wcK, bool wcQ, bool bcK, bool bcQ) {
    vector<sf::Vector2i> pseudoMoves;
    int pieceType = abs(piece);
    int color = (piece > 0) ? 1 : -1;

    auto isEnemy = [&](int r, int c){ return board[r][c] * color < 0; };
    auto isEmpty = [&](int r, int c){ return board[r][c] == 0; };
    auto isInBounds = [&](int r, int c){ return r >= 0 && r < 8 && c >= 0 && c < 8; };

    // (Copy the ENTIRE switch statement from your getLegalMoves function here)
    switch(pieceType){
        case 1: // Pawn
        {
            int dir = -color;
            if (isInBounds(row + dir, col) && isEmpty(row + dir, col)){
                pseudoMoves.push_back({col, row + dir});
                bool isStartingRank = (color == 1 && row == 6) || (color == -1 && row == 1);
                if (isStartingRank && isEmpty(row + 2 * dir, col)) {
                    pseudoMoves.push_back({col, row + 2 * dir});
                }
            }
            if (isInBounds(row + dir, col - 1) && isEnemy(row + dir, col - 1)) pseudoMoves.push_back({col - 1, row + dir});
            if (isInBounds(row + dir, col + 1) && isEnemy(row + dir, col + 1)) pseudoMoves.push_back({col + 1, row + dir});
            
            // En Passant
            if (enPassantTargetSquare.x != -1){
                if (enPassantTargetSquare.y == row + dir && abs(enPassantTargetSquare.x - col) == 1){
                    bool correctRank = (color == 1 && row == 3) || (color == -1 && row == 4);
                    if (correctRank){
                        pseudoMoves.push_back(enPassantTargetSquare);
                    }
                }
            }
            break;
        }
        case 2: // Knight
        {
            int knightMoves[8][2] = {{1, 2},{1, -2}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {-2, 1}, {-2, -1}};
            for (auto& m: knightMoves){
                int newRow = row + m[0];
                int newCol = col + m[1];
                if (isInBounds(newRow, newCol) && (isEmpty(newRow, newCol) || isEnemy(newRow, newCol))){
                    pseudoMoves.push_back({newCol, newRow});
                }
            }
            break;
        }
        case 3: // Bishop
        case 4: // Rook
        case 5: // Queen
        {
            int dirs[8][2] = {{1,0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
            int startDir = (pieceType == 4) ? 0 : (pieceType == 3) ? 4 : 0;
            int endDir = (pieceType == 4) ? 4 : (pieceType == 3) ? 8 : 8;
            for (int i = startDir; i < endDir; i++){
                for (int j = 1; j < 8; j++){
                    int newRow = row + dirs[i][0] * j;
                    int newCol = col + dirs[i][1] * j;
                    if (!isInBounds(newRow, newCol)) break;
                    if (isEmpty(newRow, newCol)){
                        pseudoMoves.push_back({newCol, newRow});
                    } else{
                        if(isEnemy(newRow, newCol)) pseudoMoves.push_back({newCol, newRow});
                        break;
                    }
                }
            }
            break;
        }
        case 6: // King
        {
            for (int i = -1; i <= 1; i++){
                for (int j = -1; j <= 1; j++){
                    if (i == 0 && j == 0) continue;
                    int newRow = row + i;
                    int newCol = col + j;
                    if (isInBounds(newRow, newCol) && (isEmpty(newRow, newCol) || isEnemy(newRow, newCol))){
                        pseudoMoves.push_back({newCol, newRow});
                    }
                }
            }

            // Castling
            if (color == 1){
                if (wcK && isEmpty(7, 5) && isEmpty(7, 6)){
                    pseudoMoves.push_back({6, 7});
                }
                if (wcQ && isEmpty(7, 1) && isEmpty(7, 2) && isEmpty(7, 3)){
                    pseudoMoves.push_back({2, 7});
                }
            } else{
                if (bcK && isEmpty(0,5) && isEmpty(0,6)){
                    pseudoMoves.push_back({6, 0});
                }
                if (bcQ && isEmpty(0, 1) && isEmpty(0, 2) && isEmpty(0, 3)){
                    pseudoMoves.push_back({2, 0});
                }
            }
            break;
        }
    }
    return pseudoMoves;
}

bool isSquareAttacked(int row, int col, int attackerColor, const int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing, const sf::Vector2i& enPassantTargetSquare, bool wcK, bool wcQ, bool bcK, bool bcQ) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] * attackerColor > 0) { // If it's an enemy piece
                // Use the new, safe function
                vector<sf::Vector2i> moves = getPseudoLegalMoves(board[r][c], r, c, board, enPassantTargetSquare, wcK, wcQ, bcK, bcQ);
                for (const auto& move : moves) {
                    if (move.x == col && move.y == row) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Function to get all legal moves for a piece
vector<sf::Vector2i> getLegalMoves(int piece, int row, int col, const int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing, const sf::Vector2i& enPassantTargetSquare, bool wcK, bool wcQ, bool bcK, bool bcQ){
    vector<sf::Vector2i> pseudoMoves = getPseudoLegalMoves(piece, row, col, board, enPassantTargetSquare, wcK, wcQ, bcK, bcQ);
    vector<sf::Vector2i> legalMoves;
    int color = (piece > 0) ? 1 : -1;
    sf::Vector2i kingPos = findKing(color, wKing, bKing);

    for (const auto& move: pseudoMoves){

        // Castling move validation
        if (abs(piece) == 6 && abs(move.x - col) == 2){
            // King cannot be in check to castle
            if (isSquareAttacked(row, col, -color, board, wKing, bKing, {-1,-1}, wcK, wcQ, bcK, bcQ)){
                continue;
            }
            // King cannot pass through an attacked square
            int passThruCol = col + (move.x > col ? 1 : -1);
            if (isSquareAttacked(row, passThruCol, -color, board, wKing, bKing, {-1, -1}, wcK, wcQ, bcK, bcQ)){
                continue;
            }
        }

        int tempBoard[8][8];
        memcpy(tempBoard, board, sizeof(int) * 64);

        // If move to the en passant square, remove the captured
        if (abs(piece) == 1 && move == enPassantTargetSquare){
            tempBoard[row][move.x] = 0; 
        }

        tempBoard[move.y][move.x] = piece;
        tempBoard[row][col] = 0;

        // Find the king's position
        sf::Vector2i newKingPos = (abs(piece) == 6) ? sf::Vector2i(move.x, move.y) : kingPos;

        // If the king is NOT attacked, it's a legal move
        if (!isSquareAttacked(newKingPos.y, newKingPos.x, -color, tempBoard, wKing, bKing, {-1, -1}, wcK, wcQ, bcK, bcQ)){
            legalMoves.push_back(move);
        }
    }
    return legalMoves;
}



// Checks if player has legal moves
bool hasLegalMoves(int playerColor, int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing, const sf::Vector2i& enPassantTargetSquare, bool wcK, bool wcQ, bool bcK, bool bcQ){
    for (int r = 0; r < 8; r++){
        for (int c = 0; c < 8; c++){
            if (board[r][c] * playerColor > 0){
                if (!getLegalMoves(board[r][c], r, c, board, wKing, bKing, enPassantTargetSquare, wcK, wcQ, bcK, bcQ).empty()){
                    return true;
                }
            }
        }
    }
    return false;
}

// Check for checkmate or stalemate
void updateGameStatus(int& currentPlayer, GameState& gameState, int& winner, int board[8][8], const sf::Vector2i wKing, const sf::Vector2i& bKing, const sf::Vector2i& enPassantTargetSquare, bool wcK, bool wcQ, bool bcK, bool bcQ, map<string, int>& history, int fiftyMoveCounter){
    
    // --- 50-Move Rule Check ---
    if (fiftyMoveCounter >= 100) { // 50 moves by each player
        gameState = GameState::GAME_OVER;
        winner = 0; // Draw by 50-move rule
        return;
    }

    currentPlayer *= -1;

    // --- Threefold Repetition Check ---
    string currentState = generateStateString(board, currentPlayer, wcK, wcQ, bcK, bcQ, enPassantTargetSquare);
    history[currentState]++;
    if (history[currentState] >= 3){
        gameState = GameState::GAME_OVER;
        winner = 0;
        return;
    }

    // Check the status of the NEW current player
    if (!hasLegalMoves(currentPlayer, board, wKing, bKing, enPassantTargetSquare, wcK, wcQ, bcK, bcQ)) {
        sf::Vector2i friendlyKingPos = findKing(currentPlayer, wKing, bKing);
        if (isSquareAttacked(friendlyKingPos.y, friendlyKingPos.x, -currentPlayer, board, wKing, bKing, enPassantTargetSquare, wcK, wcQ, bcK, bcQ)) {
            gameState = GameState::GAME_OVER;
            winner = -currentPlayer; // Checkmate, previous player wins
        }
        else {
            gameState = GameState::GAME_OVER;
            winner = 0; // Stalemate
        }
    }
}

// Converts a move to Standard Algebraic Notation (SAN)
string moveToSAN(int piece, sf::Vector2i newPos, sf::Vector2i oldPos, const int board[8][8], bool isCapture, bool isCheck, bool isCheckmate, bool isPromotion, int promotedPiece){
    stringstream san;
    int pieceType = abs(piece);
    int color = piece > 0 ? 1 : -1;

    // Castling
    if (pieceType == 6 && abs(newPos.x - oldPos.x) == 2){
        san << (newPos.x > oldPos.x ? "O-O" : "O-O-O");
    } else {
        // Piece letter (except pawns)
        if (pieceType != 1){
            map<int, char> pieceChars = {{2, 'N'}, {3, 'B'}, {4, 'R'}, {5, 'Q'}, {6, 'K'}};
            san << pieceChars[pieceType];
        }

        // Disambiguation
        if (pieceType != 1 && pieceType != 6){
            vector<sf::Vector2i> ambiguousPieces;
            for (int r = 0; r < 8; r++){
                for (int c = 0; c < 8; c++){
                    if (board[r][c] == piece && (r != oldPos.y || c != oldPos.x)){
                        vector<sf::Vector2i> moves = getLegalMoves(piece, r, c, board, {0, 0}, {0, 0}, {-1, -1}, true, true, true, true);
                        for (const auto& move: moves){
                            if (move.x == newPos.x && move.y == newPos.y){
                                ambiguousPieces.push_back({c, r});
                            }
                        }
                    }
                }
            }
            if (!ambiguousPieces.empty()){
                bool fileIsSame = false;
                for (const auto& p: ambiguousPieces){
                    if (p.x == oldPos.x) fileIsSame = true;
                }
                if (!fileIsSame){
                    san << (char)('a' + oldPos.x);
                } else{
                    san << (char)('8' - oldPos.y);
                }
            }
        }

        // Capture
        if (isCapture){
            if (pieceType == 1) san << (char)('a' + oldPos.x);
            san << 'x';
        }

        // Destination square
        san << squareToAlgebraic(newPos);

        // Promotion
        if (isPromotion){
            map<int, char> pieceChars = {{2, 'N'}, {3, 'B'}, {4, 'R'}, {5, 'Q'}};
            san << "=" << pieceChars[abs(promotedPiece)];
        }
    }

    // Check and Checkmate
    if (isCheckmate) san << "#";
    else if (isCheck) san << "+";
    
    return san.str();
}