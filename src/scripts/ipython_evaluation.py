import torch
import os

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from models.ipa2lt_head import Ipa2ltHead

# # # Setup # # #
# Parameters independent of dataset #
DEVICE = torch.device('cuda')
USE_SOFTMAX = True
MODES = ['train', 'test']
MODEL_ROOT = '../models'
LOGS_ROOT = '../logs'
AVG_BIAS_MATRICES = True

# Parameters dependent on dataset #
local_folder_root = 'train_12_21/sgd/nll'
loss = 'nll'
# phases = ['basic_only']
phases = ['no_pseudo_labeling', 'basic_only']
pretrained_model_path_no_root = '0.88565_batch64_lr0.0001311590776080191_20210102-183447_epoch1000.pt'

# label_dim = 3
# labels = ['neg', 'neutral', 'pos']
# annotator_dim = 38
# dataset = EmotionDataset(device=DEVICE)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# dataset_name = 'emotion'
# task = emotion

label_dim = 2
labels = ['neg', 'pos']
annotator_dim = 2
dataset = TripAdvisorDataset(device=DEVICE)
dataset_name = 'tripadvisor'
task = 'gender'

# label_dim = 2
# labels = ['neg', 'pos']
# annotator_dim = 2
# task = 'toxicity'
# group_by_gender = True
# only_male_female = True
# percentage = 0.05
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender,
#                            percentage=percentage, only_male_female=only_male_female)
# dataset_name = 'wikipedia'


# label_dim = 3
# labels = ['neg', 'neutral', 'pos']
# annotator_dim = 10
# padding_length = 136
# predict_coarse_attributes_task = False
# dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
#                          padding_length=padding_length)
# dataset_name = 'organic'
# task = 'sentiment'
# if predict_coarse_attributes_task:
#     task = 'coarse_attributes'
for phase in phases:
    local_folder = f'{local_folder_root}/{dataset_name}/{task}'
    model_root = f'{MODEL_ROOT}/{local_folder}'
    log_root = f'{LOGS_ROOT}/{local_folder}'
    target_model_path = f'{model_root}/{phase}'
    pretrained_model_path = f'{model_root}/pretraining/{pretrained_model_path_no_root}'

    if AVG_BIAS_MATRICES:
        avg_bias_matrices_log_path = f'{target_model_path}/avg_bias_matrices.txt'
        avg_ipa2lt_model = Ipa2ltHead(50, label_dim, annotator_dim)
        avg_amount = 0.0
        for bias_matrix in avg_ipa2lt_model.bias_matrices:
            bias_matrix.weight = torch.nn.Parameter(torch.zeros(bias_matrix.weight.shape))

    # Evaluation Loop #
    for model_path in os.listdir(target_model_path):

        # modes loop (comment out as needed)
        for mode in MODES:

            if model_path.endswith('.pt'):
                model_full_path = f'{target_model_path}/{model_path}'
                hyperparams, _ = (lambda x: (x[:-1], x[-1]))(model_path.split('.pt'))
                log_full_path = f'{target_model_path}/{hyperparams[0]}'
                if USE_SOFTMAX:
                    log_full_path += '_softmax'
                else:
                    log_full_path += '_sigmoid'
                log_full_path += f'_{mode}'
                log_full_path += '.txt'
                solver = Solver(dataset, 1e-5, 32, model_weights_path=model_full_path, device=torch.device('cuda'),
                                annotator_dim=annotator_dim, label_dim=label_dim, use_softmax=USE_SOFTMAX,
                                loss=loss)
                solver.evaluate_model(output_file_path=log_full_path, labels=labels, mode=mode,
                                      pretrained_basic_path=pretrained_model_path, basic_only=phase is 'basic_only')

                if AVG_BIAS_MATRICES and mode is MODES[0] and phase is not 'basic_only':
                    avg_amount += 1.0
                    model = Ipa2ltHead(50, label_dim, annotator_dim)
                    model.load_state_dict(torch.load(model_full_path))
                    for i in range(len(avg_ipa2lt_model.bias_matrices)):
                        avg_ipa2lt_model.bias_matrices[i].weight = torch.nn.Parameter(
                            avg_ipa2lt_model.bias_matrices[i].weight + model.bias_matrices[i].weight)

    if AVG_BIAS_MATRICES and phase is not 'basic_only':
        for bias_matrix in avg_ipa2lt_model.bias_matrices:
            bias_matrix.weight = torch.nn.Parameter(bias_matrix.weight / avg_amount)
        with open(avg_bias_matrices_log_path, 'w') as f:
            bias_out = 'Annotation bias matrices\n\n'
            for i, annotator in enumerate(dataset.annotators):
                bias_out += f'Annotator {annotator}\n'
                bias_out += f'Output\\LatentTruth'
                for label in labels:
                    bias_out += '\t' * 3 + f'{label}'
                bias_out += '\n'
                for j, label in enumerate(labels):
                    bias_out += f'{label}' + ' ' * (15 - len(label))
                    for k, label_2 in enumerate(labels):
                        bias_out += '\t' * 3 + \
                            f'{avg_ipa2lt_model.bias_matrices[i].weight[j][k].cpu().detach().numpy(): .4f}'
                    bias_out += '\n'
                bias_out += '\n'
            f.write(bias_out)
