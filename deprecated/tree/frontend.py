import pygame
import chess
import os
import threading # <-- 1. Import the threading module
from moveOrder import find_best_move_ordered

class ChessGame:
    def __init__(self, bot_depth=3):
        pygame.init()
        # Layout dimensions
        self.board_size = 640
        self.right_panel_width = 250
        self.bottom_panel_height = 80
        self.border_width = 20
        self.sq_size = self.board_size // 8
        
        self.window_width = self.board_size + self.right_panel_width + 2 * self.border_width
        self.window_height = self.board_size + self.bottom_panel_height + 2 * self.border_width
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Chess vs. Advanced Bot")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_sm = pygame.font.SysFont("Segoe UI", 20)
        self.font_md = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.timer_font = pygame.font.SysFont("Consolas", 40, bold=True)
        self.title_font = pygame.font.SysFont("Segoe UI", 72, bold=True)
        self.game_over_font = pygame.font.SysFont("Segoe UI", 52, bold=True)

        # Colors
        self.colors = {
            "bg": pygame.Color("#262421"), "light": pygame.Color("#F0D9B5"),
            "dark": pygame.Color("#B58863"), "font": pygame.Color("white"),
            "timer_low": pygame.Color("#D83B01"), "highlight_last": pygame.Color(255, 255, 51, 100),
            "highlight_legal": pygame.Color(255, 0, 0, 100)
        }

        self.bot_depth = bot_depth
        self.piece_scale_factor = 0.9
        self.load_pieces()
        self.game_state = "start_menu"
        self.start_button_rect = None
        self.restart_button_rect = None

        # Drag and drop state
        self.dragging = False; self.dragged_piece = None; self.dragged_piece_pos = None; self.drag_start_square = None
        # Animation state
        self.animating = False; self.anim_start_pos = None; self.anim_end_pos = None; self.anim_piece = None; self.anim_progress = 0.0; self.animation_speed = 4.0

        # --- 2. Add threading state variables ---
        self.bot_is_thinking = False
        self.bot_thread = None
        self.bot_move_result = None

    def reset_game(self):
        self.board = chess.Board()
        self.last_move = None; self.last_move_san = ""
        self.white_time = 18000; self.black_time = 18000
        self.bot_eval_time = 0; self.bot_positions_searched = 0
        self.game_state = "playing"

    def load_pieces(self):
            """
            Loads high-quality piece images from the 'assets' folder.
            Expects files to be named like 'wP.png', 'bK.png', etc.
            """
            self.pieces = {}
            piece_size = int(self.sq_size * self.piece_scale_factor)
            
            # Map python-chess piece types and colors to characters for filenames
            piece_type_map = {
                chess.PAWN: 'p', chess.ROOK: 'r', chess.KNIGHT: 'n',
                chess.BISHOP: 'b', chess.QUEEN: 'q', chess.KING: 'k'
            }
            color_map = {chess.WHITE: 'w', chess.BLACK: 'b'}

            for p_type in piece_type_map:
                for color in color_map:
                    # Construct the filename, e.g., 'w' + 'P' + '.png' -> 'wP.png'
                    # Note: The cburnett set uses lowercase for piece type in filename
                    filename = f"{color_map[color]}{piece_type_map[p_type].upper()}.svg"
                    filepath = os.path.join("./assets", filename)
                    
                    try:
                        img = pygame.image.load(filepath).convert_alpha()
                        # Scale the high-res image down to the desired size
                        self.pieces[(p_type, color)] = pygame.transform.smoothscale(img, (piece_size, piece_size))
                    except pygame.error as e:
                        print(f"Error loading piece image: {filepath}")
                        print(e)
                        # As a fallback, you could draw a simple shape or exit
                        # For now, we'll just print the error and continue

    # ... (get_square_from_mouse, get_pos_from_square, handle_mouse_down/motion/up are unchanged) ...
    def get_square_from_mouse(self, pos):
        x, y = pos; x -= self.border_width; y -= self.border_width
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            c = x // self.sq_size; r = 7 - (y // self.sq_size)
            return chess.square(c, r)
        return None

    def get_pos_from_square(self, square):
        r, c = divmod(square, 8); r = 7 - r
        padding = (self.sq_size * (1 - self.piece_scale_factor)) / 2
        return (self.border_width + c * self.sq_size + padding, self.border_width + r * self.sq_size + padding)

    def handle_mouse_down(self, event):
        if self.game_state != 'playing' or self.board.turn != chess.WHITE or self.animating or self.bot_is_thinking:
            return
        clicked_square = self.get_square_from_mouse(event.pos)
        if clicked_square is not None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == chess.WHITE:
                self.dragging = True
                self.drag_start_square = clicked_square
                self.dragged_piece = self.pieces[(piece.piece_type, piece.color)]
                self.dragged_piece_pos = (event.pos[0] - self.sq_size / 2, event.pos[1] - self.sq_size / 2)

    def handle_mouse_motion(self, event):
        if self.dragging:
            self.dragged_piece_pos = (event.pos[0] - self.sq_size / 2, event.pos[1] - self.sq_size / 2)

    def handle_mouse_up(self, event):
        if self.dragging:
            self.dragging = False
            end_square = self.get_square_from_mouse(event.pos)
            if end_square is not None and self.drag_start_square is not None:
                move = chess.Move(self.drag_start_square, end_square)
                if self.board.piece_type_at(self.drag_start_square) == chess.PAWN and chess.square_rank(end_square) in [0, 7]:
                    move.promotion = chess.QUEEN
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.last_move = move
            self.dragged_piece = None
            self.drag_start_square = None

    def start_bot_move_animation(self, move):
        self.animating = True
        self.anim_start_pos = self.get_pos_from_square(move.from_square)
        self.anim_end_pos = self.get_pos_from_square(move.to_square)
        piece = self.board.piece_at(move.from_square)
        self.anim_piece = self.pieces[(piece.piece_type, piece.color)]
        self.anim_progress = 0.0

    def update_animation(self, delta_time):
        if self.animating:
            self.anim_progress += self.animation_speed * delta_time
            if self.anim_progress >= 1.0:
                self.animating = False
                self.board.push(self.last_move)

    def draw_board_and_pieces(self):
        for r in range(8):
            for c in range(8):
                color = self.colors["light"] if (r + c) % 2 == 0 else self.colors["dark"]
                pygame.draw.rect(self.screen, color, (self.border_width + c * self.sq_size, self.border_width + r * self.sq_size, self.sq_size, self.sq_size))
        self.draw_highlights()
        padding = (self.sq_size * (1 - self.piece_scale_factor)) / 2
        for square, piece in self.board.piece_map().items():
            if self.dragging and square == self.drag_start_square: continue
            if self.animating and square == self.last_move.from_square: continue
            if self.animating and square == self.last_move.to_square: continue
            r, c = divmod(square, 8); r = 7 - r
            self.screen.blit(self.pieces[(piece.piece_type, piece.color)], (self.border_width + c * self.sq_size + padding, self.border_width + r * self.sq_size + padding))
        if self.dragging and self.dragged_piece:
            self.screen.blit(self.dragged_piece, self.dragged_piece_pos)
        if self.animating and self.anim_piece:
            x = self.anim_start_pos[0] + (self.anim_end_pos[0] - self.anim_start_pos[0]) * self.anim_progress
            y = self.anim_start_pos[1] + (self.anim_end_pos[1] - self.anim_start_pos[1]) * self.anim_progress
            self.screen.blit(self.anim_piece, (x, y))

    def draw_highlights(self):
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                self.draw_square_highlight(sq, self.colors["highlight_last"])
        if self.dragging:
            for move in self.board.legal_moves:
                if move.from_square == self.drag_start_square:
                    self.draw_square_highlight(move.to_square, self.colors["highlight_legal"])

    def draw_square_highlight(self, square, color):
        r, c = divmod(square, 8); r = 7 - r
        s = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, (self.border_width + c * self.sq_size, self.border_width + r * self.sq_size))

    # ... (draw_start_menu, draw_game_over, format_time, draw_right_panel, draw_bottom_panel are unchanged) ...
    def draw_start_menu(self):
        self.screen.fill(self.colors["bg"])
        title = self.title_font.render("Chess", True, self.colors["light"])
        self.screen.blit(title, title.get_rect(center=(self.window_width / 2, self.window_height / 2 - 100)))
        button_rect = pygame.Rect(0, 0, 220, 70)
        button_rect.center = (self.window_width / 2, self.window_height / 2 + 50)
        pygame.draw.rect(self.screen, self.colors["light"], button_rect, border_radius=10)
        text = self.font_md.render("Start Game", True, self.colors["bg"])
        self.screen.blit(text, text.get_rect(center=button_rect.center))
        self.start_button_rect = button_rect

    def draw_game_over(self):
        text = ""
        if self.white_time <= 0: text = "Black wins on time!"
        elif self.black_time <= 0: text = "White wins on time!"
        elif self.board.is_checkmate(): text = f"Checkmate! {'White' if self.board.result() == '1-0' else 'Black'} wins."
        else: text = "Draw!"
        overlay = pygame.Surface((self.board_size, self.board_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (self.border_width, self.border_width))
        text_surf = self.game_over_font.render(text, True, pygame.Color("white"))
        self.screen.blit(text_surf, text_surf.get_rect(center=(self.board_size / 2 + self.border_width, self.board_size / 2 + self.border_width)))
        button_rect = pygame.Rect(0, 0, 200, 60)
        button_rect.center = (self.board_size / 2 + self.border_width, self.board_size / 2 + 100 + self.border_width)
        pygame.draw.rect(self.screen, pygame.Color("white"), button_rect, border_radius=10)
        text = self.font_md.render("Main Menu", True, self.colors["bg"])
        self.screen.blit(text, text.get_rect(center=button_rect.center))
        self.restart_button_rect = button_rect

    def format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def draw_right_panel(self):
        panel_x = self.board_size + 2 * self.border_width
        turn_text = "White's Turn" if self.board.turn == chess.WHITE else "Black's Turn"
        if self.bot_is_thinking: turn_text = "Bot is thinking..."
        text_surf = self.font_md.render(turn_text, True, self.colors["font"])
        self.screen.blit(text_surf, (panel_x + 20, self.border_width + 20))
        pygame.draw.rect(self.screen, self.colors["light"], (panel_x + 20, 80, self.right_panel_width - 40, 80), border_radius=5)
        white_time_str = self.format_time(self.white_time)
        white_color = self.colors["timer_low"] if self.white_time < 30 else self.colors["bg"]
        white_surf = self.timer_font.render(white_time_str, True, white_color)
        self.screen.blit(white_surf, white_surf.get_rect(center=(panel_x + self.right_panel_width / 2, 120)))
        pygame.draw.rect(self.screen, self.colors["dark"], (panel_x + 20, self.board_size + self.border_width - 80, self.right_panel_width - 40, 80), border_radius=5)
        black_time_str = self.format_time(self.black_time)
        black_color = self.colors["timer_low"] if self.black_time < 30 else self.colors["font"]
        black_surf = self.timer_font.render(black_time_str, True, black_color)
        self.screen.blit(black_surf, black_surf.get_rect(center=(panel_x + self.right_panel_width / 2, self.board_size + self.border_width - 40)))

    def draw_bottom_panel(self):
        panel_y = self.board_size + 2 * self.border_width
        if self.last_move_san:
            text = f"Bot moved {self.last_move_san} in {self.bot_eval_time:.2f}s ({self.bot_positions_searched:,} pos)"
            text_surf = self.font_sm.render(text, True, self.colors["font"])
            self.screen.blit(text_surf, (self.border_width + 10, panel_y + 10))

    def bot_search_thread_target(self):
        """The function that will be run in the background thread."""
        # The board must be copied because the search function modifies it.
        move, t, pos = find_best_move_ordered(self.board.copy(), self.bot_depth)
        self.bot_move_result = (move, t, pos)
        

    def main_loop(self):
        running = True
        while running:
            delta_time = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_state == 'start_menu':
                        if self.start_button_rect and self.start_button_rect.collidepoint(event.pos): self.reset_game()
                    elif self.game_state == 'game_over':
                        if self.restart_button_rect and self.restart_button_rect.collidepoint(event.pos): self.game_state = 'start_menu'
                    else: self.handle_mouse_down(event)
                elif event.type == pygame.MOUSEMOTION: self.handle_mouse_motion(event)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: self.handle_mouse_up(event)

            self.screen.fill(self.colors["bg"])

            if self.game_state == 'start_menu':
                self.draw_start_menu()
            else:
                if self.game_state == 'playing':
                    # --- 3. Modified Bot Turn and Timer Logic ---
                    if not self.animating and not self.bot_is_thinking:
                        if self.board.turn == chess.WHITE: self.white_time -= delta_time
                        else: self.black_time -= delta_time

                    # Start the bot's search in a new thread
                    if self.board.turn == chess.BLACK and not self.animating and not self.bot_is_thinking and not self.board.is_game_over():
                        self.bot_is_thinking = True
                        self.bot_thread = threading.Thread(target=self.bot_search_thread_target)
                        self.bot_thread.start()
                    
                    # Check if the bot's search is finished
                    if self.bot_is_thinking and not self.bot_thread.is_alive():
                        self.bot_is_thinking = False
                        move, t, pos = self.bot_move_result
                        self.bot_move_result = None # Clear result
                        if move:
                            self.last_move_san = self.board.san(move)
                            self.last_move = move
                            self.bot_eval_time = t
                            self.bot_positions_searched = pos
                            self.black_time -= t # Subtract only the thinking time
                            self.start_bot_move_animation(move)

                    self.update_animation(delta_time)
                    if self.board.is_game_over() or self.white_time <= 0 or self.black_time <= 0:
                        self.game_state = 'game_over'
                
                self.draw_board_and_pieces()
                self.draw_right_panel()
                self.draw_bottom_panel()
                
                if self.game_state == 'game_over':
                    self.draw_game_over()
            
            pygame.display.flip()
        pygame.quit()


