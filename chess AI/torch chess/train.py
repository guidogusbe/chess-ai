import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
from auxiliary_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel
import pickle




def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

files = [file for file in os.listdir("./data") if file.endswith(".pgn")]
LIMIT_OF_FILES = min(len(files), 28)
games = []
i = 1
print(f"Files trovati: {len(files)}")
for file in tqdm(files, total=len(files)):
    print(f"Caricando file: {file}")
    games.extend(load_pgn(f"./data/{file}"))
    if i > LIMIT_OF_FILES:  # Controlla prima di caricare il file
        break
    games.extend(load_pgn(f"./data/{file}"))
    i += 1








X, y = create_input_for_nn(games)

X = X[0:2500000]
y = y[0:2500000]

y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# Create Dataset and DataLoader
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



num_epochs = 250
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60
    print(f'Epoch {epoch + 1 + 50}/{num_epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')




# Save the model
model_dir = "C:/Users/gordo/OneDrive/Documenti/1240720224744-Desktop/Desktop/chess project/models"
os.makedirs(model_dir, exist_ok=True)  # Crea la cartella se non esiste

torch.save(model.state_dict(), os.path.join(model_dir, "TORCH_100EPOCHS.pth"))
print("Modello salvato in:", os.path.join(model_dir, "TORCH_100EPOCHS.pth"))

with open(os.path.join(model_dir, "heavy_move_to_int.pkl"), "wb") as file:
    pickle.dump(move_to_int, file)
print("Mapping salvato in:", os.path.join(model_dir, "heavy_move_to_int.pkl"))