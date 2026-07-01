# predict.py - Versione corretta (senza salvataggio inutile)
from chess import Board
from auxiliary_func import board_to_matrix
import torch
from model import ChessModel
import pickle
import numpy as np
import torch.nn as nn

def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

# Carica il mapping delle mosse
with open("C:\\Users\\guido\\Desktop\\chess AI\\models\\heavy_move_to_int.pkl", "rb") as file:
    move_to_int = pickle.load(file)

# Configura dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(num_classes=len(move_to_int)).to(device)

# Caricamento sicuro del modello
checkpoint = torch.load("C:\\Users\\guido\\Desktop\\chess AI\\models\\TORCH_100EPOCHS.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

int_to_move = {v: k for k, v in move_to_int.items()}

def predict_move(board: Board):
    # Preparazione input e prediction
    X_tensor = prepare_input(board).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    
    # Estrazione mossa migliore
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    legal_moves = [move.uci() for move in board.legal_moves]
    
    return max(
        ((move, probabilities[move_to_int[move]]) for move in legal_moves),
        key=lambda x: x[1],
        default=(None, 0)
    )[0]

# Esempio di utilizzo
if __name__ == "__main__":
    board = Board()
    best_move = predict_move(board)
    if best_move:
        board.push_uci(best_move)
        print(f"Mossa effettuata: {best_move}")