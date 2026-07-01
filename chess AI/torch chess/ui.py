import pygame
import chess
from chess import Board
from predict import predict_move

# Configuration
WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
FONT_SIZE = int(SQUARE_SIZE * 0.6)

# Correct colors (inverted from the previous version)
WHITE_SQUARE = (118, 118, 118)  # Light squares
BLACK_SQUARE = (238, 238, 210)  # Dark squares
HIGHLIGHT_COLOR = (186, 202, 68)
WHITE_PIECE = (0, 0, 0)         # White pieces
BLACK_PIECE = (255, 255, 255)   # Black pieces

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI")
font = pygame.font.SysFont('Arial', FONT_SIZE, bold=True)

# Piece letter mapping
piece_letters = {
    'R': 'R', 'N': 'N', 'B': 'B', 'Q': 'Q', 'K': 'K', 'P': 'P',
    'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k', 'p': 'p'
}

def draw_board(board, selected_square=None):
    """Draw the chessboard with the correct orientation."""
    for row in range(8):
        for col in range(8):
            # Calculate coordinates by flipping the rows
            x = col * SQUARE_SIZE
            y = (7 - row) * SQUARE_SIZE
            square_color = WHITE_SQUARE if (row + col) % 2 == 0 else BLACK_SQUARE

            # Highlight the selected square
            current_square = chess.square(col, row)
            if selected_square == current_square:
                square_color = HIGHLIGHT_COLOR

            # Draw the square
            pygame.draw.rect(screen, square_color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

            # Draw the piece using the correct colors
            piece = board.piece_at(current_square)
            if piece:
                text_color = WHITE_PIECE if piece.color == chess.BLACK else BLACK_PIECE
                letter = piece_letters[piece.symbol()]

                text_surface = font.render(letter, True, text_color)
                text_rect = text_surface.get_rect(
                    center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
                )
                screen.blit(text_surface, text_rect)

    # Highlight all legal moves for the selected piece
    if selected_square is not None and board.piece_at(selected_square):
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_col = chess.square_file(move.to_square)
                to_row = 7 - chess.square_rank(move.to_square)
                x = to_col * SQUARE_SIZE + SQUARE_SIZE // 2
                y = to_row * SQUARE_SIZE + SQUARE_SIZE // 2
                pygame.draw.circle(
                    screen,
                    HIGHLIGHT_COLOR,
                    (x, y),
                    SQUARE_SIZE // 8
                )


def get_square_from_pos(pos):
    """Convert mouse coordinates to a chessboard square."""
    x, y = pos
    col = min(max(x // SQUARE_SIZE, 0), 7)
    row = 7 - min(max(y // SQUARE_SIZE, 0), 7)
    return chess.square(col, row)


def game_loop():
    board = Board()
    selected_square = None
    running = True
    ai_thinking = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle human player input (White)
            if not ai_thinking and board.turn == chess.WHITE:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    square = get_square_from_pos(event.pos)

                    if selected_square is not None:
                        # If another white piece is clicked, update the selection
                        if (
                            board.piece_at(square)
                            and board.piece_at(square).color == chess.WHITE
                        ):
                            selected_square = square
                        else:
                            # Try to execute the move
                            move = chess.Move(selected_square, square)

                            # Automatically promote pawns to queens
                            if (
                                board.piece_at(selected_square)
                                and board.piece_at(selected_square).piece_type == chess.PAWN
                            ):
                                target_rank = chess.square_rank(square)
                                if target_rank == 7 or target_rank == 0:
                                    move = chess.Move(
                                        selected_square,
                                        square,
                                        promotion=chess.QUEEN
                                    )

                            if move in board.legal_moves:
                                board.push(move)
                                ai_thinking = True

                            selected_square = None
                    else:
                        # Select a white piece
                        if (
                            board.piece_at(square)
                            and board.piece_at(square).color == chess.WHITE
                        ):
                            selected_square = square

        # AI turn (Black)
        if ai_thinking and not board.is_game_over():
            try:
                ai_move = predict_move(board)
                if ai_move:
                    board.push_uci(ai_move)
                ai_thinking = False
            except Exception as e:
                print(f"AI Error: {e}")
                ai_thinking = False

        # Render everything
        screen.fill((0, 0, 0))
        draw_board(board, selected_square)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    game_loop()