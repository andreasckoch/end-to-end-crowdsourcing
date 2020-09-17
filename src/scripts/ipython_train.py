from ..datasets.tripadvisor import TripAdvisorDataset
from ..solver import Solver
import torch

# Testing pipeline
device = torch.device('cuda')
dataset = TripAdvisorDataset(device='cuda')
solver = Solver(dataset, 1e-5, 32, device=device)

solver.fit(1)
