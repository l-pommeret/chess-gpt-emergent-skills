import chess
import chess.engine
import chess.pgn
from datetime import datetime
import tqdm
import random
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_single_game(engine_path, noise_percentage):
    """Fonction pour générer une seule partie d'échecs avec Stockfish et du bruit."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # Configurer Stockfish pour utiliser un seul thread et un niveau de compétence ajusté pour environ 1800 Elo
    engine.configure({
        "Threads": 12,
        "Skill Level": 15,  # Ajuster le Skill Level à environ 1800 Elo
    })
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        if random.random() < noise_percentage / 100:
            move = random.choice(list(board.legal_moves))
        else:
            result = engine.play(board, chess.engine.Limit(depth=5))
            move = result.move
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = board.result()  # Set the result of the game in the headers
    engine.quit()
    return game

def game_to_pgn_line(game):
    """Convertit une partie en une seule ligne PGN."""
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_str = game.accept(exporter)
    pgn_str = pgn_str.replace('\n', ' ')

    # Coller les numéros de coups avec les coups et séparer les coups blancs et noirs par un espace
    formatted_moves = []
    moves = pgn_str.split(' ')
    current_move = ''
    for move in moves:
        if move and (move[0].isdigit() and move.endswith('.')):
            if current_move:
                formatted_moves.append(current_move.strip())
                current_move = ''
            current_move += move
        else:
            current_move += move + ' '

    if current_move:
        formatted_moves.append(current_move.strip())

    pgn_line = ';' + ' '.join(formatted_moves).strip()

    # Remplacer "1/2-1/2" par "0.5-0.5"
    if game.headers["Result"] == "1/2-1/2":
        pgn_line = pgn_line.replace("1/2-1/2", "0.5-0.5")

    # Remove the last two characters (space and asterisk) if present
    if pgn_line.endswith(' *'):
        pgn_line = pgn_line[:-2]

    return pgn_line

def generate_and_format_game(args):
    """Wrapper pour générer et formater une partie."""
    engine_path, noise_percentage = args
    game = generate_single_game(engine_path, noise_percentage)
    return game_to_pgn_line(game)

def generate_chess_games(num_games, engine_path, noise_percentage):
    """Fonction pour générer plusieurs parties d'échecs en parallèle."""
    games = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_and_format_game, (engine_path, noise_percentage)) for _ in range(num_games)]

        for future in tqdm.tqdm(as_completed(futures), total=num_games, desc="Generating games"):
            game = future.result()
            games.append(game)

    return games

