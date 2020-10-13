import torch
from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset

# Setup
DEVICE = torch.device('cuda')
model_weights_path = '../models/train_10_07/full_training_pseudo_labels/_epoch1000.pt'
output_file_path = '../logs/train_10_07/full_training_pseudo_labels/evaluation_sample.txt'

dataset = TripAdvisorDataset(device=DEVICE)
solver = Solver(dataset, 1e-5, 32, model_weights_path=model_weights_path, device=torch.device('cuda'))

solver.evaluate_model(output_file_path=output_file_path)
