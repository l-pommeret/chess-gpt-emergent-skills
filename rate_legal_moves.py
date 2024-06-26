import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import chess
import random
from tqdm import tqdm

# Configuration du modèle
MODEL_DIR = "models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving model to device: {DEVICE}")

# Charger le modèle
def get_transformer_model(model_name: str, n_layers: int, device: torch.device) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=32,
        n_ctx=1023,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}{model_name}.pth", map_location=device))
    model.to(device)
    return model

# Utiliser les métadonnées fournies
meta = {
    'stoi': {' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21, 'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30, 'x': 31},
    'itos': {0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q', 22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f', 29: 'g', 30: 'h', 31: 'x'}
}

stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# Fonction pour prédire le prochain coup complet
def predict_next_move(pgn: str, model: HookedTransformer) -> str:
    if not pgn.endswith(" "):
        pgn += " "
    encoded_pgn = encode(pgn)
    input_tensor = torch.tensor(encoded_pgn).unsqueeze(0).to(DEVICE)  # Ajouter une dimension batch
    model.eval()
    move = ""

    while True:
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]  # Prédictions pour le prochain token
        next_token = torch.argmax(next_token_logits).item()
        next_char = itos[next_token]
        move += next_char
        if next_char == " ":
            break
        encoded_pgn.append(next_token)
        input_tensor = torch.tensor(encoded_pgn).unsqueeze(0).to(DEVICE)

    return move.strip()

# Fonction pour générer un état de l'échiquier à partir d'un PGN
def generate_board_from_pgn(pgn: str) -> chess.Board:
    board = chess.Board()
    moves = pgn.split(";")[1:]  # Ignorer le point-virgule au début du PGN

    for move_sequence in moves:
        individual_moves = move_sequence.split()
        for move in individual_moves:
            if "." in move:
                # Extrait le coup blanc après le point
                white_move = move.split(".")[-1]
                board.push_san(white_move)
            else:
                # Coup noir
                black_move = move
                board.push_san(black_move)

    return board

# Générer des débuts de partie aléatoires avec numéros de coups correctement placés
def generate_random_opening(max_moves: int = 2) -> str:
    while True:
        board = chess.Board()
        pgn = ";"
        move_number = 1

        for i in range(max_moves // 2):
            if board.turn == chess.WHITE:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                san_move = board.san(move)
                board.push(move)
                pgn += f"{move_number}.{san_move} "

            if board.turn == chess.BLACK:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                san_move = board.san(move)
                board.push(move)
                pgn += f"{san_move} "

            move_number += 1

        # Vérifie si le PGN contient "#"
        if "#" not in pgn:
            break

    return pgn.strip()


# Fonction pour valider les coups d'échecs en utilisant l'état de l'échiquier
def is_valid_move(board: chess.Board, move: str) -> bool:
    try:
        # Supprimer les numéros de coups et les points uniquement
        move_san = ''.join([c for i, c in enumerate(move) if c != '.' and (not c.isdigit() or (i > 0 and move[i-1].isalpha()))])
        print(f"Validating move: {move_san}")
        move = board.parse_san(move_san)
        board.push(move)
        return True
    except Exception as e:
        print(f"Erreur en validant le coup '{move_san}': {e}")
        return False

# Fonction pour tester la validité des prédictions du modèle
def test_model_predictions(n_noise: int, n_layers: int, num_samples: int) -> float:
    model_name = f"tf_lens_ckpt_{n_noise}n8l240000iter"
    model = get_transformer_model(model_name, n_layers, DEVICE)
    valid_moves = 0

    for i in tqdm(range(num_samples), desc="Testing Model Predictions"):
        random_pgn = generate_random_opening()
        if not random_pgn:
            continue
        print(f"Testing with opening: {random_pgn}")
        move = predict_next_move(random_pgn, model)

        # Générer l'état de l'échiquier à partir du PGN
        board = generate_board_from_pgn(random_pgn)

        # Valider le coup prédit
        is_valid = is_valid_move(board, move)
        valid_moves += is_valid
        print(f"PGN: {random_pgn}, Predicted move: {move}, Valid: {is_valid}")

    valid_move_rate = valid_moves / num_samples
    print(f"\nTaux de coups légaux: {valid_move_rate:.2%}")
    return valid_move_rate

    import matplotlib.pyplot as plt

# Liste des bruits à tester
noises = [0, 5, 50, 80]
n_layers = 8

# Fonction de test pour un bruit donné
def test_model(n_noise, num_samples):
    print(f"Testing {n_noise} noise with {num_samples} samples")
    valid_move_rate = test_model_predictions(n_noise, n_layers, num_samples)
    return n_noise, valid_move_rate

# Exécuter les tests de manière séquentielle
results = {}
for n_noise in noises:
    if n_noise in [0, 5]:
        num_samples = 3000
    else:
        num_samples = 2000
    n_noise, valid_move_rate = test_model(n_noise, num_samples)
    results[n_noise] = valid_move_rate

# Afficher les résultats sous forme de tableau
print("Résultats bruts :")
print("------------------------------")
print("| Taux de bruit | Taux de coups légaux |")
print("------------------------------")
for n_noise, valid_move_rate in results.items():
    print(f"| {n_noise:13} | {valid_move_rate:21.2%} |")
print("------------------------------")


# Tracer le graphique bruit/taux de coups légaux
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', color='b')
plt.xlabel('Taux de bruit')
plt.ylabel('Taux de coups légaux')
plt.title('Taux de coups légaux en fonction du bruit')
plt.grid(True)
plt.show()
