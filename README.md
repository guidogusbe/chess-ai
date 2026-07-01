```markdown
# Deep Learning Chess AI

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Pygame](https://img.shields.io/badge/pygame-2.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the source code for a custom Chess Artificial Intelligence driven by a Convolutional Neural Network (CNN) implemented in PyTorch. The project integrates a graphical user interface (GUI) built with Pygame, allowing human players to test the model's capabilities in real-time. The neural network is designed to predict the optimal move probability distribution based on the current board state, having been trained on a dataset of high-level chess matches.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Deep Learning Model](#deep-learning-model)
4. [Data Pipeline](#data-pipeline)
5. [Graphical User Interface](#graphical-user-interface)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Roadmap and Future Enhancements](#roadmap-and-future-enhancements)

---

## Architecture Overview

The system is strictly divided into three primary components:
* **Data Processing:** Parsing Portable Game Notation (PGN) files into a numerical format suitable for matrix multiplication.
* **Model Training & Inference:** A PyTorch-based CNN that ingests a 13-channel tensor representation of the chessboard and outputs a classification over all possible legal chess moves.
* **User Interface:** A Pygame front-end that handles user input, strictly enforces chess rules via the `python-chess` library, and visually renders the board state and AI responses.

---

## Directory Structure

The repository is structured to separate data, serialized models, and executable code.

```text
chess ai/
├── data/
│   └── lichess_elite_2021-12.pgn    # Raw dataset of high-Elo chess matches in PGN format
├── models/
│   ├── heavy_move_to_int.pkl        # Serialized dictionary mapping UCI move strings to integer classes
│   └── TORCH_100EPOCHS.pth          # PyTorch state_dict containing the trained model weights
└── torch chess/
    ├── __pycache__/                 # Compiled Python bytecode
    ├── auxiliary_func.py            # Utility functions for board-to-tensor encoding and decoding
    ├── dataset.py                   # PyTorch Dataset subclass for batching and preprocessing PGN data
    ├── model.py                     # CNN architecture definition (PyTorch nn.Module)
    ├── predict.py                   # Inference logic to load weights and evaluate the board state
    ├── train.py                     # Training loop, loss calculation, and backpropagation logic
    └── ui.py                        # Pygame rendering engine and main event loop

```

---

## Deep Learning Model

The core neural network is defined in `torch chess/model.py`. It utilizes a sequential convolutional architecture tailored for spatial feature extraction on an 8x8 grid.

### Input Representation

The chessboard is encoded into a `(13, 8, 8)` float tensor:

* **Channels 0-5:** White pieces (Pawn, Knight, Bishop, Rook, Queen, King).
* **Channels 6-11:** Black pieces (Pawn, Knight, Bishop, Rook, Queen, King).
* **Channel 12:** Encodes auxiliary board state information (e.g., player turn, castling rights, or legal move masks).

### Network Topology

1. **Conv Block 1:** `nn.Conv2d` (In: 13, Out: 64, Kernel: 3x3, Padding: 1) followed by a ReLU activation. The padding ensures the spatial dimensions remain 8x8, focusing on local piece interactions.
2. **Conv Block 2:** `nn.Conv2d` (In: 64, Out: 128, Kernel: 3x3, Padding: 1) followed by a ReLU activation. This layer extracts higher-level tactical motifs.
3. **Flattening:** The multidimensional tensor is flattened into a 1D vector of size `8 * 8 * 128` (8192 elements).
4. **Fully Connected Block:** A dense `nn.Linear` layer mapping 8192 inputs to 256 hidden units, followed by a ReLU activation.
5. **Output Layer:** A final `nn.Linear` layer mapping the 256 hidden units to `num_classes`. The output consists of raw, unnormalized logits representing every unique move in the training dataset.

### Weight Initialization

To prevent vanishing or exploding gradients and ensure rapid convergence during the initial epochs:

* Convolutional layers use **Kaiming Uniform** initialization (optimized for ReLU).
* Linear layers use **Xavier Uniform** initialization.

---

## Data Pipeline

The files `dataset.py` and `auxiliary_func.py` handle the transformation of raw data into training inputs.

1. **PGN Parsing:** The dataset (`lichess_elite_2021-12.pgn`) is read incrementally. The `python-chess` library simulates the games move-by-move.
2. **Move Encoding:** Each unique move (in UCI format, e.g., `e2e4`) is assigned a unique integer identifier. This mapping is saved as a pickle file (`heavy_move_to_int.pkl`) to ensure consistency between training and inference phases.
3. **Tensor Generation:** For each board state, `auxiliary_func.py` generates the 13-channel tensor and pairs it with the integer target of the move actually played by the human expert.

---

## Graphical User Interface

The application GUI (`ui.py`) provides a robust environment to play against the trained model.

* **Rendering:** The board is drawn using standard matrix coordinates, flipped vertically to ensure White is at the bottom. Square and piece sizes scale dynamically based on configuration variables.
* **Move Validation:** All moves are strictly validated against `board.legal_moves`.
* **Visual Indicators:** Clicking a piece highlights its square and draws target circles on all valid destination squares.
* **Pawn Promotion:** Handled automatically; pawns reaching the 8th or 1st rank are instantly promoted to Queens to streamline UI interactions.
* **Inference Integration:** When the human player (White) completes a valid move, the game state triggers `predict_move(board)` from `predict.py`. The AI (Black) calculates its move, which is then parsed from UCI format and applied to the board.

---

## Installation

### Prerequisites

* Python 3.8 or higher.
* A CUDA-capable GPU is highly recommended for both training and inference, though the CPU is supported.

### Setup Instructions

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/chess-ai-pytorch.git](https://github.com/YOUR-USERNAME/chess-ai-pytorch.git)
cd chess-ai-pytorch

```


2. **Initialize a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows environments use: venv\Scripts\activate

```


3. **Install required packages:**
```bash
pip install torch torchvision torchaudio
pip install python-chess pygame

```



---

## Usage

### Running the Game (Inference Mode)

To launch the GUI and play against the pre-trained model:

1. Ensure `TORCH_100EPOCHS.pth` and `heavy_move_to_int.pkl` are present in the `models/` directory.
2. Execute the UI script:
```bash
cd "torch chess"
python ui.py

```



*Note: The AI's calculation phase runs on the main thread and may temporarily block the UI depending on your hardware specifications.*

### Training a New Model

To train the CNN from scratch or fine-tune it on a different dataset:

1. Place your uncompressed `.pgn` file in the `data/` directory.
2. Update the file paths in `train.py` if necessary.
3. Execute the training loop:
```bash
cd "torch chess"
python train.py

```



This script will iterate through the dataset, calculate the Cross-Entropy Loss, backpropagate gradients, and save updated `.pth` weights to the `models/` folder at defined epoch intervals.

---

## Roadmap and Future Enhancements

The current implementation serves as a foundational supervised learning model. Future iterations will focus on the following upgrades:

* **Monte Carlo Tree Search (MCTS):** Transitioning from a pure policy network to an AlphaZero-style architecture by combining the CNN's policy output with MCTS for deeper lookahead and tactical verification.
* **Value Head Integration:** Expanding `model.py` to output a scalar evaluation of the board state (e.g., continuous value between -1.0 and 1.0) alongside the move probabilities.
* **Asynchronous Inference:** Decoupling the `predict_move` execution from the Pygame event loop using Python's `threading` or `asyncio` to prevent the application from freezing during calculation times.
* **ONNX Export:** Converting the PyTorch `.pth` model to an ONNX graph to significantly reduce inference overhead during gameplay.
* **Advanced GUI Features:** Adding algebraic notation logging, match clocks, and handling edge-case draw rules (threefold repetition, 50-move rule).

---

*Distributed under the MIT License. See LICENSE for more information.*

```

```
