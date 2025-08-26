#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <map>
using namespace std;

// Helper to load a texture and handle errors
bool loadPieceTexture(map<int, sf::Texture>& textures, int piece, const string& filename){
    if(!textures[piece].loadFromFile(filename)){
        cerr << "Failed to load" << filename << endl;
        return false;
    }
    return true;
}

// Helpet function to load the board state from a FEN string
void loadBoardFromFen(int board[8][8], const string& fen, sf::Vector2i& wKing, sf::Vector2i& bKing){
    for(int i = 0; i < 8; i++) for(int j = 0; j < 8; j++) board[i][j] = 0;

    map<char, int> pieceMap = {
        {'P', 1}, {'N', 2}, {'B', 3}, {'R', 4}, {'Q', 5}, {'K', 6},
        {'p', -1}, {'n', -2}, {'b', -3}, {'r', -4}, {'q', -5}, {'k', -6}
    };

    int row = 0, col = 0;
    for (char c: fen){
        if (c == ' '){
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
}

// Finds the king of a given color on the board.
sf::Vector2i findKing(int color, const sf::Vector2i& wKing, const sf::Vector2i& bKing) {
    return (color == 1) ? wKing : bKing;
}

vector<sf::Vector2i> getPseudoLegalMoves(int piece, int row, int col, const int board[8][8]) {
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
            break;
        }
    }
    return pseudoMoves;
}

bool isSquareAttacked(int row, int col, int attackerColor, const int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] * attackerColor > 0) { // If it's an enemy piece
                // Use the new, safe function
                vector<sf::Vector2i> moves = getPseudoLegalMoves(board[r][c], r, c, board);
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
vector<sf::Vector2i> getLegalMoves(int piece, int row, int col, const int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing, bool checkValidation = true){
    vector<sf::Vector2i> pseudoMoves = getPseudoLegalMoves(piece, row, col, board);
    vector<sf::Vector2i> legalMoves;
    int color = (piece > 0) ? 1 : -1;
    sf::Vector2i kingPos = findKing(color, wKing, bKing);

    for (const auto& move: pseudoMoves){
        int tempBoard[8][8];
        memcpy(tempBoard, board, sizeof(int) * 64);
        tempBoard[move.y][move.x] = piece;
        tempBoard[row][col] = 0;

        // Find the king's position
        sf::Vector2i newKingPos = (abs(piece) == 6) ? sf::Vector2i(move.x, move.y) : kingPos;

        // If the king is NOT attacked, it's a legal move
        if (!isSquareAttacked(newKingPos.y, newKingPos.x, -color, tempBoard, wKing, bKing)){
            legalMoves.push_back(move);
        }
    }
    return legalMoves;
}



// Checks if player has legal moves
bool hasLegalMoves(int playerColor, int board[8][8], const sf::Vector2i& wKing, const sf::Vector2i& bKing){
    for (int r = 0; r < 8; r++){
        for (int c = 0; c < 8; c++){
            if (board[r][c] * playerColor > 0){
                if (!getLegalMoves(board[r][c], r, c, board, wKing, bKing).empty()){
                    return true;
                }
            }
        }
    }
    return false;
}

int main()
{
    const int tileSize = 80, N = 8;
    const int boardPixels = tileSize * 8;

    // Window
    sf::RenderWindow window(sf::VideoMode({(unsigned)boardPixels, (unsigned)boardPixels}), "SFML Chess");
    window.setFramerateLimit(60);

    // King 
    sf::Vector2i whiteKingPos;
    sf::Vector2i blackKingPos;

    // Board
    int board[N][N] = {0};
    const string startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    loadBoardFromFen(board, startFEN, whiteKingPos, blackKingPos);

    // Piece
    map<int, sf::Texture> pieceTextures; 
    loadPieceTexture(pieceTextures, 1, "resource/wP.png");
    loadPieceTexture(pieceTextures, 2, "resource/wN.png");
    loadPieceTexture(pieceTextures, 3, "resource/wB.png");
    loadPieceTexture(pieceTextures, 4, "resource/wR.png");
    loadPieceTexture(pieceTextures, 5, "resource/wQ.png");
    loadPieceTexture(pieceTextures, 6, "resource/wK.png");
    loadPieceTexture(pieceTextures, -1, "resource/bP.png");
    loadPieceTexture(pieceTextures, -2, "resource/bN.png");
    loadPieceTexture(pieceTextures, -3, "resource/bB.png");
    loadPieceTexture(pieceTextures, -4, "resource/bR.png");
    loadPieceTexture(pieceTextures, -5, "resource/bQ.png");
    loadPieceTexture(pieceTextures, -6, "resource/bK.png");

    // Sprite
    sf::RectangleShape tile({(float)tileSize, (float)tileSize});
    sf::Sprite pieceSprite(pieceTextures[1]);
    sf::RectangleShape moveIndicator({(float)tileSize, (float)tileSize});
    moveIndicator.setFillColor(sf::Color(180, 0, 0, 120));
    const int textureTileSize = 213;

    // Logic
    int currentPlayer = 1; // 1 = White, -1 = Black
    bool isMoving = false;
    int selectedPiece = 0;
    sf::Vector2i oldPos;
    vector<sf::Vector2i> legalMoves;
    bool gameOver = false;
    int winner = 0; // 1 for white, -1 for black
    

    // Check & Checkmate
    sf::RectangleShape checkIndicator({(float)tileSize, (float)tileSize});
    checkIndicator.setFillColor(sf::Color(255, 0, 0, 80));
    sf::Font font("resource/arial.ttf");
    sf::Text popupText(font, "");
    popupText.setFont(font);
    popupText.setCharacterSize(40);
    popupText.setFillColor(sf::Color::White);

    sf::RectangleShape popupBackground(sf::Vector2f(400, 200));
    popupBackground.setFillColor(sf::Color(0, 0, 0, 200));
    popupBackground.setOutlineColor(sf::Color::White);
    popupBackground.setOutlineThickness(2);
    popupBackground.setPosition(sf::Vector2{boardPixels / 2.f - 200, boardPixels / 2.f - 100});

    sf::RectangleShape closeButton(sf::Vector2f(120, 50));
    closeButton.setFillColor(sf::Color(100, 100, 100));
    closeButton.setPosition(sf::Vector2{boardPixels / 2.f - 60, boardPixels / 2.f + 20});

    sf::Text closeButtonText(font, "close", 24);
    closeButtonText.setPosition(sf::Vector2{boardPixels / 2.f - 40, boardPixels / 2.f + 30});

    while (window.isOpen()){
        // Get Mouse Pos
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);

        // --- Event ---
        while (auto event = window.pollEvent()){

            // --- CLOSE EVENT --- 
            if(event->is<sf::Event::Closed>()){
                window.close();
            }
            
            // --- MOUSE PRESSED EVENT ---
            if (event->getIf<sf::Event::MouseButtonPressed>()){
                int col = mousePos.x / tileSize;
                int row = mousePos.y / tileSize;

                // Make sure the click is within the board
                if (col >= 0 && col < N && row >= 0 && row < N){

                    // If Game Over
                    if (gameOver && closeButton.getGlobalBounds().contains(sf::Vector2{(float)mousePos.x, (float)mousePos.y})){
                        window.close();
                    }

                    int piece = board[row][col];
                    if (piece != 0 && (piece * currentPlayer > 0)){
                        isMoving = true;
                        selectedPiece = piece;
                        oldPos = {col, row};
                        board[row][col] = 0;
                        legalMoves = getLegalMoves(selectedPiece, row, col, board, whiteKingPos, blackKingPos);
                    }
                }
            }

            // --- MOUSE BUTTON RELEASED ---
            if (auto* mouse = event->getIf<sf::Event::MouseButtonReleased>()){
                if (mouse->button == sf::Mouse::Button::Left){
                    if (isMoving){
                        isMoving = false;
                        // Calculate which square the piece was dropped on
                        int col = mousePos.x / tileSize;
                        int row = mousePos.y / tileSize;

                        sf::Vector2i newPos = {col, row};
                        bool moveIsLegal = false;

                        // Check if move is legal
                        for (const auto& move: legalMoves){
                            if(move.x == newPos.x && move.y == newPos.y){
                                moveIsLegal = true;
                                break;
                            }
                        }

                        // Place the piece
                        if (moveIsLegal){
                            if(abs(selectedPiece) == 6){
                                (selectedPiece > 0) ? whiteKingPos = newPos : blackKingPos = newPos;
                            }
                            board[row][col] = selectedPiece;

                            int pieceType = abs(selectedPiece);
                            if (pieceType == 1){
                                if ((selectedPiece > 0 && row == 0) || (selectedPiece < 0 && row == 7)){
                                    board[row][col] = 5 * currentPlayer;
                                }
                            }

                            currentPlayer *= -1;
                            sf::Vector2i opponentKingPos = findKing(currentPlayer, whiteKingPos, blackKingPos);
                            if (isSquareAttacked(opponentKingPos.y, opponentKingPos.x, -currentPlayer, board, whiteKingPos, blackKingPos)){
                                
                                // Checkmate
                                if(!hasLegalMoves(currentPlayer, board, whiteKingPos, blackKingPos)){
                                    gameOver = true;
                                    winner = -currentPlayer;
                                } 
                            } else{
                                
                                // Stalemate
                                if (!hasLegalMoves(currentPlayer, board, whiteKingPos, blackKingPos)){
                                    gameOver = true;
                                    winner = 0; 
                                }
                            }
                        }
                        else{
                            board[oldPos.y][oldPos.x] = selectedPiece;
                        }
                        legalMoves.clear();
                    }
                }
            }
        }

        window.clear(sf::Color::Black); // background

        // --- Draw Board ---
        for (int y = 0; y < N; y++){
            for (int x= 0; x < N; x++){
                tile.setPosition(sf::Vector2f{x * (float)tileSize, y * (float)tileSize});
                if ((x + y) % 2 == 0) {
                    tile.setFillColor(sf::Color(240, 217, 181)); // light
                } else {
                    tile.setFillColor(sf::Color(181, 136, 99));  // dark
                }                
                window.draw(tile);
            }
        }

        // --- Draw legal move indicators ---
        for (const auto& move: legalMoves){
            moveIndicator.setPosition(sf::Vector2f{
                move.x * (float)tileSize, move.y * (float)tileSize
            });
            window.draw(moveIndicator);
        }

        // --- Draw pieces ---
        for (int y = 0; y < N; y++){
            for (int x = 0; x < N; x++){
                int piece = board[y][x];
                if (piece != 0){
                    // set the correct texture from map
                    pieceSprite.setTexture(pieceTextures.at(piece));

                    sf::Vector2u textureSize = pieceTextures.at(piece).getSize();
                    pieceSprite.setScale(sf::Vector2{
                        (float)tileSize / textureSize.x,
                        (float)tileSize / textureSize.y
                    });
                    // Position the piece on the board
                    pieceSprite.setPosition(sf::Vector2{x * (float)tileSize, y * (float)tileSize});
                }

                window.draw(pieceSprite);
            }
        }

        // --- Draw check indicator ---
        sf::Vector2i friendlyKingPos = findKing(currentPlayer, whiteKingPos, blackKingPos);
        if (isSquareAttacked(friendlyKingPos.y, friendlyKingPos.x, -currentPlayer, board, whiteKingPos, blackKingPos)) {
            checkIndicator.setPosition(sf::Vector2{friendlyKingPos.x * (float)tileSize, friendlyKingPos.y * (float)tileSize});
            window.draw(checkIndicator);
        }

        // --- Draw Game Over Popup ---
        if (gameOver) {
            // Draw semi-transparent overlay
            sf::RectangleShape overlay({(float)boardPixels, (float)boardPixels});
            overlay.setFillColor(sf::Color(0, 0, 0, 150));
            window.draw(overlay);

            // Set text and draw popup
            string winnerText;
            if (winner == 0) winnerText = "Draw by Stalemate!";
            else winnerText = (winner == 1) ? "White wins by Checkmate!" : "Black wins by Checkmate!";
            popupText.setString(winnerText);
            popupText.setPosition(sf::Vector2{boardPixels / 2.f - popupText.getGlobalBounds().size.x / 2.f, boardPixels / 2.f - 70});
            
            window.draw(popupBackground);
            window.draw(popupText);
            window.draw(closeButton);
            window.draw(closeButtonText);
        }

        // --- Draw moving piece ---
        if (isMoving)
        {
            pieceSprite.setTexture(pieceTextures.at(selectedPiece));
            sf::Vector2u textureSize = pieceTextures.at(selectedPiece).getSize();
            pieceSprite.setScale(sf::Vector2{(float)tileSize / textureSize.x, (float)tileSize / textureSize.y});
            pieceSprite.setPosition(sf::Vector2{(float)mousePos.x - tileSize / 2.f, (float)mousePos.y - tileSize / 2.f});
            window.draw(pieceSprite);
        }
        
        window.display();
    }
}