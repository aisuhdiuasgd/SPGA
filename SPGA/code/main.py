
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch import nn

import utils
from dataloader import GetData
from model import MDI, SPGA

SEED = 2021
utils.set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LATENT_DIM = 32
LR = 1e-3
EPOCHS = 500
TRAIN_SAMPLES_PER_EPOCH = 2000

DATA_PATH = os.getenv("SPGA_DATA_PATH", r"D:\data\RNADisease")

def build_node_features(dataset: GetData) -> np.ndarray:

    train_matrix = np.zeros((471, 84), dtype=np.float32)
    d_feature = pd.read_csv("D:/d_feature.csv", index_col=0)
    m_feature = pd.read_csv("D:/m_feature.csv", index_col=0)
    mat1 = np.hstack((d_feature, train_matrix))
    mat2 = np.hstack((train_matrix.T, m_feature))
    feature = np.vstack((mat1, mat2)).astype(np.float32)

    return feature


def train_one_epoch(dataset: GetData, model: SPGA, optimiser: torch.optim.Optimizer):

    model.train()

    X = build_node_features(dataset)

    triplets = utils.sample_uniform_triplets(dataset, n_samples=TRAIN_SAMPLES_PER_EPOCH)

    if triplets.size == 0:
        print("Warning: No triplets sampled. Skipping epoch.")
        return

    s = triplets[:, 0]
    d_pos = triplets[:, 1]
    d_neg = triplets[:, 2]

    sno_batch = np.concatenate([s, s], axis=0)
    dis_batch = np.concatenate([d_pos, d_neg], axis=0)
    labels = np.concatenate([np.ones_like(s), np.zeros_like(s)], axis=0).astype(np.float32)

    preds = model(sno_batch, dis_batch, X)  # (B,)
    loss_fn = nn.BCELoss()
    loss = loss_fn(preds, torch.as_tensor(labels, device=preds.device))

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

    return loss.item()


def main():

    file_no = "1"

    print(f"Loading data for fold {file_no}...")
    dataset = GetData(file_no, path=DATA_PATH)
    print("Data loaded successfully.")

    print("Initializing model and optimizer...")
    model = SPGA(MDI(LATENT_DIM), dataset, latent_dim=LATENT_DIM, dropout_bool=False).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    print("Model and optimizer are ready.")

    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(dataset, model, optimiser)
        if epoch % 10 == 0:
            if loss is not None:
                print(f"Epoch: {epoch:03d}/{EPOCHS} | Loss: {loss:.4f}")
            else:
                print(f"Epoch: {epoch:03d}/{EPOCHS} | No triplets sampled.")

    print("\n--- Training Finished ---")


if __name__ == "__main__":
    main()