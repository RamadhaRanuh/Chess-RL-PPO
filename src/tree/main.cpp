#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include "chess.hpp"
#include "minimax.hpp"
using namespace std;

int main()
{

    Zobrist::init();

    const int tileSize = 80, N = 8;
    const int boardPixels = tileSize * 8;
    const int rightPanelWidth = 320;
    const int windowWidth = boardPixels + rightPanelWidth;
    const int windowHeight = boardPixels;

    // Window
    sf::RenderWindow window(sf::VideoMode({(unsigned)windowWidth, (unsigned)windowHeight}), "SFML Chess");
    window.setFramerateLimit(60);

    // King 
    sf::Vector2i whiteKingPos, blackKingPos;

    // Castling
    bool whiteCanCastleKingSide = true, whiteCanCastleQueenSide = true, blackCanCastleKingSide = true, blackCanCastleQueenSide = true;

    // Board
    int board[N][N];
    const string startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    loadBoardFromFen(board, startFEN, whiteKingPos, blackKingPos, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide);

    // Piece
    map<int, sf::Texture> pieceTextures; 
    loadPieceTexture(pieceTextures, 1, "resource/wP.png"); loadPieceTexture(pieceTextures, 2, "resource/wN.png"); loadPieceTexture(pieceTextures, 3, "resource/wB.png"); loadPieceTexture(pieceTextures, 4, "resource/wR.png"); loadPieceTexture(pieceTextures, 5, "resource/wQ.png"); loadPieceTexture(pieceTextures, 6, "resource/wK.png");
    loadPieceTexture(pieceTextures, -1, "resource/bP.png"); loadPieceTexture(pieceTextures, -2, "resource/bN.png"); loadPieceTexture(pieceTextures, -3, "resource/bB.png"); loadPieceTexture(pieceTextures, -4, "resource/bR.png"); loadPieceTexture(pieceTextures, -5, "resource/bQ.png"); loadPieceTexture(pieceTextures, -6, "resource/bK.png");

    // Sprite
    sf::RectangleShape tile({(float)tileSize, (float)tileSize});
    sf::Sprite pieceSprite(pieceTextures[1]);
    sf::RectangleShape moveIndicator({(float)tileSize, (float)tileSize});
    moveIndicator.setFillColor(sf::Color(180, 0, 0, 120));

    // Logic
    int currentPlayer = 1; // 1 = White, -1 = Black
    bool isMoving = false;
    int selectedPiece = 0;
    sf::Vector2i oldPos;
    vector<sf::Vector2i> legalMoves;
    GameState gameState = GameState::CHOOSING_COLOR;
    sf::Vector2i enPassantTargetSquare = {-1, -1};
    sf::Vector2i promotionSquare = {-1, -1};
    int winner = 0; // 1 for white, -1 for black
    map<string, int> positionHistory; // Threefold repetition rule
    int fiftyMoveCounter = 0; // 50-move rule
    vector<string> moveHistory; // FEN History

    int playerColor = 1;

    unsigned long long aiCalculations = 0;
    

    // Check & Checkmate
    sf::RectangleShape checkIndicator({(float)tileSize, (float)tileSize});
    checkIndicator.setFillColor(sf::Color(255, 0, 0, 80));
    sf::Font font("resource/arial.ttf");
    sf::Text popupText(font, "");
    popupText.setFont(font);
    popupText.setCharacterSize(40);
    popupText.setFillColor(sf::Color::White);

    // --- MODIFIED --- Setup for AI Info Text
    sf::Text aiInfoText(font, "");
    aiInfoText.setFont(font);
    aiInfoText.setCharacterSize(18);
    aiInfoText.setFillColor(sf::Color(180, 180, 180));
    aiInfoText.setPosition(sf::Vector2f{boardPixels + 15, 400}); // Position it below move history

    sf::Text moveHistoryText(font, "");
    moveHistoryText.setFont(font);
    moveHistoryText.setCharacterSize(20);
    moveHistoryText.setFillColor(sf::Color::White);
    moveHistoryText.setPosition(sf::Vector2f{boardPixels + 15, 10});

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

    // --- NEW --- UI Elements for Color Choice Screen
    sf::Text choiceText(font, "Play as:", 50); choiceText.setPosition(sf::Vector2{windowWidth / 2.f - 100, windowHeight / 2.f - 150});
    sf::RectangleShape whiteButton(sf::Vector2f(200, 80)); whiteButton.setFillColor(sf::Color(240, 217, 181)); whiteButton.setPosition(sf::Vector2{windowWidth / 2.f - 100, windowHeight / 2.f - 50});
    sf::Text whiteButtonText(font, "White", 40); whiteButtonText.setFillColor(sf::Color::Black); whiteButtonText.setPosition(sf::Vector2{windowWidth / 2.f - 50, windowHeight / 2.f - 35});
    sf::RectangleShape blackButton(sf::Vector2f(200, 80)); blackButton.setFillColor(sf::Color(100, 100, 100)); blackButton.setPosition(sf::Vector2{windowWidth / 2.f - 100, windowHeight / 2.f + 50});
    sf::Text blackButtonText(font, "Black", 40); blackButtonText.setFillColor(sf::Color::White); blackButtonText.setPosition(sf::Vector2{windowWidth / 2.f - 50, windowHeight / 2.f + 65});


    while (window.isOpen()){
        // Get Mouse Pos
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);

        // --- Event ---
        while (auto event = window.pollEvent()){

            // --- CLOSE EVENT --- 
            if(event->is<sf::Event::Closed>()) window.close();
            
            // --- Event handling for color choice screen ---
            if (gameState == CHOOSING_COLOR) {
                if (auto* mouse = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (whiteButton.getGlobalBounds().contains(sf::Vector2f(mousePos.x, mousePos.y))) {
                        playerColor = 1;
                        gameState = GameState::PLAYING;
                    } else if (blackButton.getGlobalBounds().contains(sf::Vector2f(mousePos.x, mousePos.y))) {
                        playerColor = -1;
                        gameState = GameState::PLAYING;
                    }
                }
            }
            
            // --- IF PLAYING ---
            else if (gameState == PLAYING){
                // --- MOUSE PRESSED EVENT ---
                if (currentPlayer == playerColor) {
                    if (event->getIf<sf::Event::MouseButtonPressed>()){
                        int col = mousePos.x / tileSize; int row = mousePos.y / tileSize;

                        // Make sure the click is within the board
                        if (col >= 0 && col < N && row >= 0 && row < N){
                            int piece = board[row][col];
                            if (piece != 0 && (piece * currentPlayer > 0)){
                                isMoving = true;
                                selectedPiece = piece;
                                oldPos = {col, row};
                                legalMoves = getLegalMoves(selectedPiece, row, col, board, whiteKingPos, blackKingPos, enPassantTargetSquare, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide);
                            }
                        }
                    }
                    if (auto* mouse = event->getIf<sf::Event::MouseButtonReleased>()){
                        if (mouse->button == sf::Mouse::Button::Left){
                            if (isMoving){
                                isMoving = false;
                                int col = mousePos.x / tileSize; int row = mousePos.y / tileSize; sf::Vector2i newPos = {col, row};
                                bool moveIsLegal = false;
                                for (const auto& move: legalMoves) if(move.x == newPos.x && move.y == newPos.y){ moveIsLegal = true; break; }
                                if (moveIsLegal){
                                    makeMove({oldPos, newPos}, board, currentPlayer, gameState, winner, whiteKingPos, blackKingPos, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, enPassantTargetSquare, positionHistory, fiftyMoveCounter, moveHistory, promotionSquare);
                                }
                                legalMoves.clear();
                            }
                        }
                    }
                }
            } 
            else if (gameState == PROMOTING){
                if (auto* mouse = event->getIf<sf::Event::MouseButtonPressed>()){
                    if (mouse->button == sf::Mouse::Button::Left){
                        int clickedRow = mousePos.y / tileSize;
                        if (clickedRow == promotionSquare.y){
                            int clickedCol = mousePos.x / tileSize; int choice = 0;
                            if (clickedCol == 2) choice = 5; else if (clickedCol == 3) choice = 4; else if (clickedCol == 4) choice = 3; else if (clickedCol == 5) choice = 2;
                            if (choice != 0){
                                int promotingColor = (currentPlayer > 0) ? 1 : -1;
                                board[promotionSquare.y][promotionSquare.x] = choice * promotingColor;
                                string lastMove = moveHistory.back(); moveHistory.pop_back(); map<int, char> pieceChars = {{2, 'N'}, {3, 'B'}, {4, 'R'}, {5, 'Q'}}; lastMove.insert(lastMove.length() - (lastMove.back() == '+' ? 1 : 0), "=" + string(1, pieceChars[choice])); moveHistory.push_back(lastMove);
                                gameState = GameState::PLAYING; enPassantTargetSquare = {-1, -1};
                                updateGameStatus(currentPlayer, gameState, winner, board, whiteKingPos, blackKingPos, enPassantTargetSquare, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, positionHistory, fiftyMoveCounter);
                            }
                        }
                    }
                }
            }  
            
            // --- IF GAME OVER ---
            else if(gameState == GAME_OVER){
                if (event->getIf<sf::Event::MouseButtonPressed>()) if (closeButton.getGlobalBounds().contains(sf::Vector2f(mousePos.x, mousePos.y))) window.close();
            }
        }

        if (gameState == PLAYING && currentPlayer != playerColor) {
            Move aiMove = getAIBestMove(board, currentPlayer, whiteKingPos, blackKingPos, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, enPassantTargetSquare, aiCalculations);
            makeMove(aiMove, board, currentPlayer, gameState, winner, whiteKingPos, blackKingPos, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, enPassantTargetSquare, positionHistory, fiftyMoveCounter, moveHistory, promotionSquare);
        }

        window.clear(sf::Color(30, 30, 30)); // background

        // --- NEW --- Drawing logic for different game states
        if (gameState == CHOOSING_COLOR) {
            window.draw(choiceText);
            window.draw(whiteButton);
            window.draw(whiteButtonText);
            window.draw(blackButton);
            window.draw(blackButtonText);
        } else {
            for (int y = 0; y < N; y++) for (int x= 0; x < N; x++){
                tile.setPosition(sf::Vector2f{x * (float)tileSize, y * (float)tileSize});
                if ((x + y) % 2 == 0) tile.setFillColor(sf::Color(240, 217, 181)); else tile.setFillColor(sf::Color(181, 136, 99));                
                window.draw(tile);
            }
            for (const auto& move: legalMoves){ moveIndicator.setPosition(sf::Vector2f{move.x * (float)tileSize, move.y * (float)tileSize}); window.draw(moveIndicator); }
            for (int y = 0; y < N; y++) for (int x = 0; x < N; x++){
                int piece = board[y][x];
                if (piece != 0){
                    pieceSprite.setTexture(pieceTextures.at(piece));
                    sf::Vector2u textureSize = pieceTextures.at(piece).getSize();
                    pieceSprite.setScale(sf::Vector2{(float)tileSize / textureSize.x, (float)tileSize / textureSize.y});
                    pieceSprite.setPosition(sf::Vector2{x * (float)tileSize, y * (float)tileSize});
                    window.draw(pieceSprite);
                }
            }
            sf::Vector2i friendlyKingPos = findKing(currentPlayer, whiteKingPos, blackKingPos);
            if (isSquareAttacked(friendlyKingPos.y, friendlyKingPos.x, -currentPlayer, board, whiteKingPos, blackKingPos, enPassantTargetSquare, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide)) {
                checkIndicator.setPosition(sf::Vector2{friendlyKingPos.x * (float)tileSize, friendlyKingPos.y * (float)tileSize});
                window.draw(checkIndicator);
            }
            stringstream historyStream;
            for (size_t i = 0; i < moveHistory.size(); ++i) {
                if (i % 2 == 0) historyStream << (i / 2) + 1 << ". " << moveHistory[i] << " ";
                else historyStream << moveHistory[i] << "\n";
            }
            moveHistoryText.setString(historyStream.str());
            window.draw(moveHistoryText);

            stringstream aiInfoStream;
            aiInfoStream << "AI Info:\n";
            aiInfoStream << "Depth: " << AI_SEARCH_DEPTH << "\n";
            aiInfoStream << "Positions: " << aiCalculations;
            aiInfoText.setString(aiInfoStream.str());
            window.draw(aiInfoText);


            if (gameState == GAME_OVER) {
                sf::RectangleShape overlay({(float)windowWidth, (float)windowHeight}); overlay.setFillColor(sf::Color(0, 0, 0, 150)); window.draw(overlay);
                string winnerText;
                if (winner == 0) {
                    string finalState = generateStateString(board, currentPlayer, whiteCanCastleKingSide, whiteCanCastleQueenSide, blackCanCastleKingSide, blackCanCastleQueenSide, enPassantTargetSquare);
                    if (fiftyMoveCounter >= 100) winnerText = "Draw by 50-Move Rule!"; else if (positionHistory[finalState] >= 3) winnerText = "Draw by Repetition!"; else winnerText = "Draw by Stalemate!";
                } else {
                    winnerText = (winner == 1) ? "White wins by Checkmate!" : "Black wins by Checkmate!";
                    if (!moveHistory.empty() && moveHistory.back().find('#') == string::npos) moveHistory.back() += '#';
                }
                popupText.setString(winnerText); popupText.setPosition(sf::Vector2{boardPixels / 2.f - popupText.getGlobalBounds().size.x / 2.f, boardPixels / 2.f - 70});
                window.draw(popupBackground); window.draw(popupText); window.draw(closeButton); window.draw(closeButtonText);
            }
            if (gameState == PROMOTING){
                sf::RectangleShape overlay({(float)boardPixels, (float)boardPixels}); overlay.setFillColor(sf::Color(0, 0, 0, 100)); window.draw(overlay);
                int color = (currentPlayer > 0) ? 1 : -1; int pieceToDraw[] = {5, 4, 3, 2};
                for (int i = 0; i < 4; i++){
                    pieceSprite.setTexture(pieceTextures.at(pieceToDraw[i] * color));
                    pieceSprite.setPosition(sf::Vector2f{(2 + i) * (float)tileSize, promotionSquare.y * (float)tileSize});
                    window.draw(pieceSprite);
                }
            }
            if (isMoving) {
                pieceSprite.setTexture(pieceTextures.at(selectedPiece));
                sf::Vector2u textureSize = pieceTextures.at(selectedPiece).getSize();
                pieceSprite.setScale(sf::Vector2{(float)tileSize / textureSize.x, (float)tileSize / textureSize.y});
                pieceSprite.setPosition(sf::Vector2{(float)mousePos.x - tileSize / 2.f, (float)mousePos.y - tileSize / 2.f});
                window.draw(pieceSprite);
            }
        }
        
        window.display();
    }
}

