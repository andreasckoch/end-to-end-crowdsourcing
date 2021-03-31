import torch
import pickle
import os

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from models.ipa2lt_head import Ipa2ltHead
from utils import get_best_model_path

# # # Setup # # #
# Parameters independent of dataset #
DEVICE = torch.device('cuda')
USE_SOFTMAX = True
MODES = ['test', 'validation']
MODEL_ROOT = '../models'
LOGS_ROOT = '../logs'
AVERAGING_METHOD = 'macro'

# Parameters dependent on dataset #
local_folder_root = 'train_02_14/sgd/nll'
loss = 'nll'
# phases = ['basic_only']
phases = ['ltnet_true', 'ltnet_pseudo', 'basic_true', 'basic_pseudo', 'dawid_skene', 'mace']
# phases = ['dawid_skene', 'mace']
label_maps = ['dawid_skene', 'majority_voting']
# pretrained_model_path_no_root = '0.88711_batch64_lr4.897628213733225e-05_20210127-101008_epoch200.pt'

# label_dim = 3
# annotator_dim = 38
# loss = 'nll'
# dataset_name = 'emotion'
# domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
# dataset = EmotionDataset(device=DEVICE, domain_embedding_path=domain_embedding_path)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# task = emotion
# if domain_embedding_path is not '':
#     task += '_fine_tuned_emb'
# epoch_factor = 10
# eval_setting = 'label_map'

label_dim = 2
annotator_dim = 2
labels = ['neg', 'pos']
one_dataset_one_annotator = False
dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator)
dataset_name = 'tripadvisor'
task = 'gender'
if one_dataset_one_annotator:
    task = 'dataset'
eval_setting = 'label_map'

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
# eval_setting = 'label_map'


# label_dim = 3
# labels = ['neg', 'neutral', 'pos']
# annotator_dim = 10
# padding_length = 136
# predict_coarse_attributes_task = False
# dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
#                          padding_length=padding_length)
# dataset_name = 'organic'
# task = 'sentiment_fine_tuned_emb'
# if predict_coarse_attributes_task:
#     task = 'coarse_attributes'
# eval_setting = 'label_map'

# for the tripadvisor dataset, dawid_skene or majority voting make no sense --> evaluate on single annotators
outer_iterator = label_maps
if eval_setting == 'one_annotator':
    outer_iterator = dataset.annotators

for iter_element in outer_iterator:
    if eval_setting == 'label_map':
        label_map = iter_element
        label_map_path = f"../data/{label_map}/{dataset_name}"
        if dataset_name is 'tripadvisor':
            if one_dataset_one_annotator:
                label_map_path += '/1.3'
            else:
                label_map_path += '/1.2'
        for mode in MODES:
            labels_path = label_map_path
            labels_path += f'/sample_label_map_{mode}.pkl'

            sample_label_map = {}
            with open(labels_path, 'rb') as f:
                sample_label_map = pickle.load(f)
            dataset.use_custom_labels(sample_label_map, mode=mode)

        dataset.no_annotator_filter()

    if eval_setting == 'one_annotator':
        annotator = iter_element
        dataset.set_annotator_filter(annotator)

    for phase in phases:
        if phase in ['ltnet_true', 'ltnet_pseudo', 'basic_true', 'basic_pseudo']:
            local_folder = f'{local_folder_root}/{dataset_name}/{task}'
            model_root = f'{MODEL_ROOT}/{local_folder}'
            target_model_path = f'{model_root}/{phase}'
            pretrained_model_path = get_best_model_path(f'{model_root}/pretraining')
        if phase in ['dawid_skene', 'mace']:
            target_model_path = f'{MODEL_ROOT}/{local_folder_root}/{phase}/{dataset_name}/{task}'
            pretrained_model_path = ''

        # Prepare file to document performances
        performance_name = f'model_performances_{iter_element}.txt'
        performance_path = f'{target_model_path}/{performance_name}'
        with open(performance_path, 'w') as f:
            title = 'MODEL PERFORMANCES\n\n\n'
            f.write(title)

            # modes loop (comment out as needed)
            for mode in MODES:
                f.write(f'{mode.upper()} MODE\n\n\n')

                metrics_out = ''
                metrics = {}
                # Evaluation Loop #
                for model_path in os.listdir(target_model_path):

                    if model_path.endswith('.pt'):
                        hyperparams, _ = (lambda x: (x[:-1], x[-1]))(model_path.split('.pt'))
                        model_full_path = f'{target_model_path}/{model_path}'
                        solver = Solver(dataset, 1e-5, 32, model_weights_path=model_full_path, device=torch.device('cuda'),
                                        annotator_dim=annotator_dim, label_dim=label_dim, use_softmax=USE_SOFTMAX,
                                        loss=loss)
                        model_metrics = solver.evaluate_model_simple(pretrained_basic_path=pretrained_model_path,
                                                                     basic_only=(phase in ['basic_true', 'basic_pseudo', 'dawid_skene', 'mace']),
                                                                     mode=mode, return_metrics=True, averaging_method=AVERAGING_METHOD)
                        metrics[hyperparams[0]] = model_metrics
                        model_entry = f'Model {hyperparams[0]}: Accuracy {model_metrics[0]:.4f}, F1 {model_metrics[2]:.4f}\n\n'
                        metrics_out += model_entry
                        # f.write(model_entry)

                # find best model
                accuracies = {metric[0]: metric[1][0] for metric in metrics.items()}
                best_accuracy_model = max(accuracies, key=accuracies.get)
                f1s = {metric[0]: metric[1][2] for metric in metrics.items()}
                best_f1_model = max(f1s, key=f1s.get)
                best_out = f'Best accuracy model: {best_accuracy_model} with accuracy {accuracies[best_accuracy_model]:.4f}, \
                             f1 {f1s[best_accuracy_model]:.4f}\n\n'
                best_out += f'Best f1 model: {best_f1_model} with accuracy {accuracies[best_f1_model]:.4f}, f1 {f1s[best_f1_model]:.4f}\n\n'

                # Document pretrained model performance
                best_out += f'Pretrained model performance: accuracy {metrics[best_f1_model][1]:.4f}, f1 {metrics[best_f1_model][3]:.4f}\n\n\n\n'

                f.write(best_out)
                f.write(metrics_out)
