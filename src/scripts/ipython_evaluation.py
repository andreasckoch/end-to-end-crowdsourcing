import torch
from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset

# Setup
DEVICE = torch.device('cuda')
model_weights_path = '../models/train_09_29/full_training_little_pretraining/ipa_0.87500_batch64_lr3.253788822801386e-06_20200930-073454_epoch500.pt'
output_file_path = '../logs/train_09_29/full_training_little_pretraining/ipa_0.87500_batch64_lr3.253788822801386e-06_20200930-073454_epoch500.txt'

dataset = TripAdvisorDataset(device=DEVICE)
solver = Solver(dataset, 1e-5, 32, model_weights_path=model_weights_path, device=torch.device('cuda'))

solver.evaluate_model(output_file_path=output_file_path)
