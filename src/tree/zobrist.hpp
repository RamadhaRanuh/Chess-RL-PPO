#pragma once
#include <cstdint>
#include <SFML/Graphics.hpp>
#include <random>
#include <unordered_map> 

// Zobrist Hashing for transposition table
namespace Zobrist {
    uint64_t piece_keys[12][64]; // [piece_type][square]
    uint64_t black_to_move_key;
    uint64_t castling_keys[16]; // A bitmask from 0000 to 1111 represents all castle rights
    uint64_t en_passant_keys[8]; // One for each file

    void init() {
        std::mt19937_64 gen(1984); // A fixed seed for deterministic keys
        std::uniform_int_distribution<uint64_t> dist;
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 64; ++j) {
                piece_keys[i][j] = dist(gen);
            }
        }
        black_to_move_key = dist(gen);
        for (int i = 0; i < 16; ++i) {
            castling_keys[i] = dist(gen);
        }
        for (int i = 0; i < 8; ++i) {
            en_passant_keys[i] = dist(gen);
        }
    }

    int piece_to_index(int piece) {
        // Maps piece {-6, -5, ..., 5, 6} to index {0, 1, ..., 11}
        return piece + 6 - (piece > 0 ? 1 : 0);
    }

    uint64_t generate_hash(const int board[8][8], int current_player, bool wcK, bool wcQ, bool bcK, bool bcQ, const sf::Vector2i& ep_square) {
        uint64_t hash = 0;
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                if (board[r][c] != 0) {
                    hash ^= piece_keys[piece_to_index(board[r][c])][r * 8 + c];
                }
            }
        }
        if (current_player == -1) {
            hash ^= black_to_move_key;
        }
        int castle_rights = (wcK << 3) | (wcQ << 2) | (bcK << 1) | bcQ;
        hash ^= castling_keys[castle_rights];

        if (ep_square.x != -1) {
            hash ^= en_passant_keys[ep_square.x];
        }
        return hash;
    }
}

// Transposition Table Entry
struct TTEntry {
    int score;
    int depth;
    enum { EXACT, LOWER_BOUND, UPPER_BOUND } flag;
};

// The table itself
std::unordered_map<uint64_t, TTEntry> transposition_table;