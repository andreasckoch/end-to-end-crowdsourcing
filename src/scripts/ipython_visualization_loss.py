from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import tensorflow as tf
import torch
import pytz
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

DEVICE = torch.device('cuda')
ROOT = '../logs'
LOCAL_FOLDER = 'train_02_14/sgd/nll'
VISUALS_FOLDER = '../visuals/loss'
COLORS = {
    'ltnet_true': ['midnightblue', 'steelblue', 'lightsteelblue'],
    'ltnet_pseudo': ['darkgreen', 'forestgreen', 'limegreen'],
    'basic_true': ['firebrick', 'indianred', 'lightcoral'],
    'basic_pseudo': ['darkorange', 'sandybrown', 'peachpuff'],
    'dawid_skene': ['lemonchiffon', 'palegoldenrod', 'goldenrod'],
    'mace': ['mediumorchid', 'mediumpurple', 'indigo'],
    # 'ltnet_true': ['steelblue', 'steelblue', 'steelblue'],
    # 'ltnet_pseudo': ['darkgreen', 'darkgreen', 'darkgreen'],
    # 'basic_true': ['firebrick', 'firebrick', 'firebrick'],
    # 'basic_pseudo': ['darkorange', 'darkorange', 'darkorange'],
    # 'dawid_skene': ['goldenrod', 'goldenrod', 'goldenrod'],
    # 'mace': ['mediumpurple', 'mediumpurple', 'mediumpurple'],
}
AVG_KERNEL_SIZE = 17

TAG_KEY = 'Loss'
NOTE = 'funny'  # included in png name

emotion_path = f'emotion/valence_fine_tuned_emb'
organic_path = f'organic/sentiment_fine_tuned_emb'
tripadvisor_path = f'tripadvisor/gender'
tripadvisor_task3_path = f'tripadvisor/dataset'

titles = {
    'ltnet_true': 'Ltnet (True)',
    'ltnet_pseudo': 'Ltnet (Pseudo)',
    'basic_true': 'Basic (True)',
    'basic_pseudo': 'Basic (Pseudo)',
    'dawid_skene': 'Dawid-Skene',
    'mace': 'MACE',
}
axes_ind = {key: i for i, key in enumerate(list(titles.keys()))}

# models to be included
model_paths = {
    # 'emotion': [
    #     'pretraining/0.43333_batch64_lr2.3022332755203014e-05_20210314-002006_epoch100.pt',
    #     'ltnet_true/0.43333_batch64_lr2.4334463927868494e-05_20210314-021045_epoch10.pt',
    #     'ltnet_true/0.32778_batch64_lr0.00019401625894506697_20210314-015553_epoch10.pt',
    #     'ltnet_true/0.27778_batch64_lr0.0004022421411892705_20210314-025634_epoch200.pt',
    #     'ltnet_pseudo/0.38889_batch64_lr7.532422652040585e-05_20210314-090150_epoch100.pt',
    #     'ltnet_pseudo/0.38889_batch64_lr6.558011114814137e-05_20210314-063216_epoch200.pt',
    #     'ltnet_pseudo/0.38889_batch64_lr0.0001738572135788551_20210314-110857_epoch100.pt',
    #     'basic_true/0.35556_batch64_lr0.00029142340518243965_20210314-133053_epoch10.pt',
    #     'basic_true/0.27778_batch64_lr0.00037822080088599464_20210314-133324_epoch300.pt',
    #     'basic_true/0.25000_batch64_lr0.0004919815631127703_20210314-133119_epoch300.pt',
    #     'basic_pseudo/0.38889_batch64_lr0.00012474202039158085_20210314-155153_epoch300.pt',
    #     'basic_pseudo/0.38889_batch64_lr1.6492166082639642e-06_20210314-155747_epoch300.pt',
    #     'basic_pseudo/0.38889_batch64_lr6.234854126567167e-05_20210314-135551_epoch300.pt',
    #     'dawid_skene/0.40000_batch64_lr0.000967200592308404_20210314-010121_epoch3000.pt',
    #     'dawid_skene/0.48333_batch64_lr7.743066309061042e-06_20210314-013647_epoch10.pt',
    #     'dawid_skene/0.38889_batch64_lr1.3605691179963449e-05_20210314-011655_epoch200.pt',
    #     # 'dawid_skene/0.37222_batch64_lr2.858573215218724e-06_20210314-005647_epoch3000.pt',
    #     'mace/0.39444_batch64_lr4.116371230787667e-06_20210314-093321_epoch100.pt',
    #     'mace/0.33889_batch64_lr1.2770701377074906e-05_20210314-095309_epoch1000.pt',
    #     'mace/0.38889_batch64_lr5.259751257068534e-06_20210314-093554_epoch2000.pt',
    # ],
    # 'organic': [
    #     'pretraining/0.39589_batch64_lr1.0502885914785642e-06_20210214-193813_epoch100.pt',
    #     'ltnet_true/0.46041_batch64_lr0.00012074746286220034_20210215-010822_epoch200.pt',
    #     'ltnet_true/0.39296_batch64_lr3.1439561211675358e-06_20210215-005459_epoch200.pt',
    #     'ltnet_true/0.38710_batch64_lr1.1647163762346628e-06_20210215-002823_epoch300.pt',
    #     'ltnet_pseudo/0.45161_batch64_lr0.0007457348732371428_20210215-030122_epoch300.pt',
    #     'ltnet_pseudo/0.39296_batch64_lr1.4136505863089356e-06_20210215-045038_epoch10.pt',
    #     'ltnet_pseudo/0.38710_batch64_lr2.107994924767966e-05_20210215-021637_epoch10.pt',
    #     'basic_true/0.39003_batch64_lr6.897383863022446e-05_20210215-092250_epoch200.pt',
    #     'basic_true/0.39003_batch64_lr1.2896509655876734e-06_20210215-091717_epoch300.pt',
    #     'basic_true/0.39003_batch64_lr9.91672634475231e-05_20210215-091322_epoch10.pt',
    #     'basic_pseudo/0.39296_batch64_lr0.0005360373675296286_20210215-124412_epoch300.pt',
    #     'basic_pseudo/0.39003_batch64_lr1.555028751471085e-06_20210215-125418_epoch10.pt',
    #     'basic_pseudo/0.39003_batch64_lr4.234618209307088e-06_20210215-120408_epoch10.pt',
    #     'dawid_skene/0.39296_batch64_lr4.283744567062306e-06_20210310-021717_epoch100.pt',
    #     'dawid_skene/0.39296_batch64_lr1.2214634104932345e-05_20210309-213758_epoch10.pt',
    #     'dawid_skene/0.39589_batch64_lr6.961751956026354e-06_20210223-110207_epoch10.pt',
    #     'mace/0.39296_batch64_lr0.0003399677575710899_20210224-101033_epoch10.pt',
    #     'mace/0.39589_batch64_lr1.2561707083071053e-05_20210224-120141_epoch10.pt',
    #     'mace/0.39003_batch64_lr1.3493334217460814e-06_20210224-110558_epoch10.pt',
    # ],
    # 'tripadvisor': [
    #     'pretraining/0.88652_batch64_lr0.0007793276871859566_20210215-164407_epoch200.pt',
    #     'ltnet_true/0.88040_batch64_lr0.00038444874246692216_20210215-173649_epoch300.pt',
    #     'ltnet_true/0.88215_batch64_lr2.809523532379282e-05_20210215-182443_epoch200.pt',
    #     'ltnet_true/0.88127_batch64_lr4.2578740523705225e-06_20210215-190848_epoch10.pt',
    #     # 'ltnet_true/0.88127_batch64_lr0.0001036484270750035_20210215-192446_epoch10.pt',
    #     # 'ltnet_true/0.88127_batch64_lr0.00016007147324825029_20210216-022533_epoch100.pt',
    #     # 'ltnet_true/0.87952_batch64_lr0.00048010960708537524_20210215-174828_epoch200.pt',
    #     'ltnet_pseudo/0.88098_batch64_lr6.336491063463214e-05_20210216-063404_epoch100.pt',
    #     'ltnet_pseudo/0.88273_batch64_lr0.00011025119684842613_20210216-061701_epoch10.pt',
    #     'ltnet_pseudo/0.88156_batch64_lr0.0005791964070970774_20210216-065122_epoch100.pt',
    #     'basic_true/0.88040_batch64_lr5.5222013599350325e-05_20210216-103223_epoch200.pt',
    #     'basic_true/0.88244_batch64_lr1.0358833078325748e-05_20210216-094153_epoch200.pt',
    #     'basic_true/0.88011_batch64_lr0.00015882038016239278_20210216-091235_epoch100.pt',
    #     'basic_pseudo/0.88069_batch64_lr2.637013559802613e-06_20210216-103926_epoch200.pt',
    #     'basic_pseudo/0.88098_batch64_lr3.3859874590406155e-05_20210216-135349_epoch10.pt',
    #     'basic_pseudo/0.88127_batch64_lr1.0968407317910714e-05_20210216-115324_epoch200.pt',
    #     'dawid_skene/0.88649_batch64_lr1.2868161655291325e-05_20210222-145913_epoch500.pt',
    #     'dawid_skene/0.89203_batch64_lr0.0002511595731753984_20210222-123704_epoch100.pt',
    #     'dawid_skene/0.87748_batch64_lr0.00040446547793737735_20210221-172227_epoch300.pt',
    #     'mace/0.87398_batch64_lr4.2140703831450715e-05_20210220-184127_epoch200.pt',
    #     'mace/0.87719_batch64_lr0.0003097014642243996_20210220-193216_epoch300.pt',
    #     'mace/0.86027_batch64_lr4.700039506450802e-05_20210221-120857_epoch100.pt',
    # ],
    'tripadvisor_task3': [
        'pretraining/0.89046_batch64_lr0.00032733913574841474_20210217-014616_epoch200.pt',
        'ltnet_true/0.89056_batch64_lr2.312452956539194e-06_20210217-043443_epoch10.pt',
        'ltnet_true/0.89004_batch64_lr7.010446531226476e-05_20210217-083212_epoch200.pt',
        'ltnet_true/0.89025_batch64_lr8.34718027268253e-05_20210217-090918_epoch300.pt',
        'ltnet_pseudo/0.88627_batch64_lr0.00012226706987482848_20210217-155908_epoch10.pt',
        'ltnet_pseudo/0.87486_batch64_lr0.0002537058003130355_20210217-151035_epoch300.pt',
        'ltnet_pseudo/0.88784_batch64_lr1.064388792279937e-05_20210217-141953_epoch100.pt',
        'basic_true/0.89015_batch64_lr0.0003977529736239607_20210218-084638_epoch100.pt',
        'basic_true/0.88878_batch64_lr4.312780716409973e-05_20210218-064557_epoch300.pt',
        'basic_true/0.88899_batch64_lr5.644756691702812e-05_20210218-062620_epoch300.pt',
        'basic_pseudo/0.88784_batch64_lr3.950367857808452e-06_20210218-165939_epoch10.pt',
        'basic_pseudo/0.88638_batch64_lr8.396566364706828e-06_20210218-162412_epoch10.pt',
        'basic_pseudo/0.88857_batch64_lr1.606391973201756e-06_20210218-094041_epoch10.pt',
        'dawid_skene/0.82983_batch64_lr0.0001557436779965454_20210313-031416_epoch500.pt',
        'dawid_skene/0.83737_batch64_lr0.0001774597938956025_20210312-134607_epoch10.pt',
        'dawid_skene/0.83894_batch64_lr0.0005004941225135539_20210312-151029_epoch500.pt',
        'mace/0.85056_batch64_lr1.5102203520225228e-05_20210311-124648_epoch500.pt',
        'mace/0.85684_batch64_lr3.2832679110856566e-05_20210311-173427_epoch500.pt',
        'mace/0.85234_batch64_lr1.0839316679090854e-05_20210311-022329_epoch500.pt',
    ],
}


# helper functions
def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def add_arrow(line, position=None, direction='right', size=15, color=None, border_size=10):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        start_position = (max(xdata) + min(xdata)) / 2
        mean_position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - start_position))
    mean_start_ind = np.argmin(np.absolute(xdata - mean_position))
    if direction == 'right':
        end_ind = start_ind + 1
        mean_end_ind = mean_start_ind + 1
    else:
        end_ind = start_ind - 1
        mean_end_ind = mean_start_ind - 1

    # average over ydata
    ydata_avg_window_1 = ydata[max(
        0, start_ind - border_size):max(1, start_ind)]
    ydata_avg_1 = sum(ydata_avg_window_1) / len(ydata_avg_window_1)
    ydata_avg_window_2 = ydata[min(
        len(ydata) - 1, end_ind):min(len(ydata), end_ind + border_size)]
    ydata_avg_2 = sum(ydata_avg_window_2) / len(ydata_avg_window_2)
    xdata_avg_window_1 = xdata[max(
        0, start_ind - border_size):max(1, start_ind)]
    xdata_avg_1 = sum(xdata_avg_window_1) / len(xdata_avg_window_1)
    xdata_avg_window_2 = xdata[min(
        len(ydata) - 1, end_ind):min(len(xdata), end_ind + border_size)]
    xdata_avg_2 = sum(xdata_avg_window_2) / len(xdata_avg_window_2)

    arrow_vector = np.array(
        [xdata_avg_2 - xdata_avg_1, ydata_avg_2 - ydata_avg_1])
    arrow_vector = arrow_vector / np.linalg.norm(arrow_vector)
    desired_size = np.linalg.norm(np.array(
        [xdata[mean_end_ind] - xdata[mean_start_ind], ydata[mean_end_ind] - ydata[mean_start_ind]]))
    arrow_vector = arrow_vector * desired_size
    # print(arrow_vector)

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[start_ind] + arrow_vector[0],
                           ydata[start_ind] + arrow_vector[1]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )


for key in list(model_paths.keys()):

    colors = COLORS.copy()

    # collect train/val loss of all models
    visuals_dir = f'{VISUALS_FOLDER}/{key}'
    model_loss = {}

    for model_path in model_paths[key]:
        sub_dir, path = (lambda x: (x[0], x[1]))(model_path.split('/'))
        hyperparams, _ = (lambda x: (x[:-1], x[-1]))(path.split('.pt'))
        f1, writer_hyperparams, epoch = (lambda x: (
            x[0], x[1:-1], x[-1]))(hyperparams[0].split('_'))

        directory = f'{ROOT}/{LOCAL_FOLDER}'
        if key == 'tripadvisor_task3':
            directory += '/1.3'
        if sub_dir in ['dawid_skene', 'mace']:
            directory += f'/{sub_dir}'
        if key == 'emotion':
            directory += f'/{emotion_path}'
        elif key == 'organic':
            directory += f'/{organic_path}'
        elif key == 'tripadvisor':
            directory += f'/{tripadvisor_path}'
        elif key == 'tripadvisor_task3':
            directory += f'/{tripadvisor_task3_path}'
        if sub_dir in ['ltnet_true', 'ltnet_pseudo', 'basic_true', 'basic_pseudo', 'pretraining']:
            directory += f'/{sub_dir}'
        sep = '_'
        writer_dir = f'{directory}/writer_{sep.join(writer_hyperparams)}'
        writer_path = f'{writer_dir}/{os.listdir(writer_dir)[0]}'

        model_loss[model_path] = {
            'train': {},
            'validation': {},
        }
        for event in my_summary_iterator(writer_path):
            for value in event.summary.value:
                if TAG_KEY in value.tag:
                    for mode in list(model_loss[model_path].keys()):
                        if mode in value.tag:
                            if event.step not in list(model_loss[model_path][mode].keys()):
                                model_loss[model_path][mode][event.step] = {}
                            _, annotator, _ = (lambda x: (x[0], x[1], x[2]))(
                                value.tag.split('/'))
                            _, annotator = (lambda x: (x[0], x[1]))(
                                annotator.split(' '))
                            model_loss[model_path][mode][event.step][annotator] = value.simple_value

        if sub_dir == 'pretraining':
            pretraining_loss = model_loss[model_path]
            for mode in list(pretraining_loss.keys()):
                mode_pretraining_loss = list(pretraining_loss[mode].values())
                mode_pretraining_loss = [list(loss_step.values())[
                    0] for loss_step in mode_pretraining_loss]
                pretraining_loss[mode] = mode_pretraining_loss
            model_loss[model_path] = pretraining_loss

    # diagram with all model's train losses drawn against validation losses to show possible overfitting
    fig, axes = plt.subplots(3, 2, figsize=(8, 7))
    plt.rcParams["font.family"] = "serif"
    fig.tight_layout(h_pad=4.0, w_pad=13.0)
    axes = axes.flatten()
    annotate_points = {}
    for i, model_path in enumerate(list(model_loss.keys())):
        sub_dir, path = (lambda x: (x[0], x[1]))(model_path.split('/'))
        hyperparams, _ = (lambda x: (x[:-1], x[-1]))(path.split('.pt'))
        lr = (lambda x: (x[2]))(hyperparams[0].split('_'))
        if lr[-4] == 'e':
            lr = lr[2:6] + lr[-4:]
        else:
            lr = lr[2:10]
        if sub_dir == 'pretraining':
            continue

        loss = model_loss[model_path]
        for mode in list(model_loss[model_path].keys()):
            mode_loss = list(loss[mode].values())
            # assuming all samples have same number of annotators
            if len(mode_loss[0]) > 1:
                # average across annotators
                mode_loss = [sum(list(loss_step.values())) /
                             len(loss_step) for loss_step in mode_loss]
            else:
                mode_loss = [list(loss_step.values())[0]
                             for loss_step in mode_loss]
            loss[mode] = mode_loss

        # merge pretraining data with all ltnet/basic models
        if sub_dir in ['ltnet_true', 'ltnet_pseudo', 'basic_true', 'basic_pseudo']:
            pretraining_model_path = list(model_loss.keys())[0]
            if (lambda x: (x[0], x[1]))(pretraining_model_path.split('/'))[0]:
                pretraining_loss = model_loss[pretraining_model_path]
                loss['train'] = pretraining_loss['train'] + loss['train']
                loss['validation'] = pretraining_loss['validation'] + \
                    loss['validation']

        # average over epochs to reduce noise
        loss_train_np = np.array(loss['train'])
        loss_validation_np = np.array(loss['validation'])
        kernel = np.ones(AVG_KERNEL_SIZE) * (1 / AVG_KERNEL_SIZE)
        loss_train_avg = np.convolve(loss_train_np, kernel, mode='same')
        loss_validation_avg = np.convolve(
            loss_validation_np, kernel, mode='same')
        border_size = int(AVG_KERNEL_SIZE / 2 - 0.5)
        loss_train_avg[:border_size] = loss_train_np[:border_size]
        loss_train_avg[- border_size:] = loss_train_np[- border_size:]
        loss_validation_avg[:border_size] = loss_validation_np[:border_size]
        loss_validation_avg[- border_size:] = loss_validation_np[- border_size:]

        if (sub_dir == 'mace' or sub_dir == 'dawid_skene') and len(loss_train_avg.tolist()) < 600:
            append_len = 600 - len(loss_train_avg.tolist())
            last_value_train = loss_train_avg[-1]
            last_value_validation = loss_validation_avg[-1]
            loss_train_avg = np.concatenate(
                (loss_train_avg, last_value_train * np.ones(append_len)))
            loss_validation_avg = np.concatenate(
                (loss_validation_avg, last_value_validation * np.ones(append_len)))

        # pop color
        color = colors[sub_dir].pop(0)

        line = axes[axes_ind[sub_dir]].plot(range(len(loss_train_avg.tolist())), loss_train_avg.tolist(), color=color,
                                            label=lr)[0]
        # line = axes[axes_ind[sub_dir]].plot(loss_validation_avg.tolist(), loss_train_avg.tolist(), color=color,
        #                                     marker='D', label=lr, markersize=1, alpha=0.5)[0]
        line2 = axes[axes_ind[sub_dir]].plot(range(len(loss_validation_avg.tolist())), loss_validation_avg.tolist(), color=color,
                                             label=lr, linestyle='dashed')[0]
        #  marker='D', label=lr, markersize=1, alpha=0.5, linestyle='dashed')[0]
        # border_size = 40
        # if sub_dir in ['ltnet_true', 'ltnet_pseudo', 'basic_true', 'basic_pseudo']:
        #     border_size = 160
        # add_arrow(line, color='black', border_size=border_size)
        # annotate_points = {
        #     'start': [(loss['validation'][0], loss['train'][0]), (0, 10)],
        #     'end': [(loss['validation'][-1], loss['train'][-1]), (-20, -10)],
        # }
        # for item in list(annotate_points.items()):
        #     plt.annotate(item[0],  # this is the text
        #                  item[1][0],  # this is the point to label
        #                  textcoords="offset points",  # how to position the text
        #                  xytext=item[1][1],  # distance from text to points (x,y)
        #                  ha='center')  # horizontal alignment can be left, right or center

        axes[axes_ind[sub_dir]].set_ylabel('Loss')
        axes[axes_ind[sub_dir]].set_xlabel('Epochs')
        # axes[axes_ind[sub_dir]].set_ylabel('Training Loss')
        # axes[axes_ind[sub_dir]].set_xlabel('Validation Loss')
        axes[axes_ind[sub_dir]].set_title(f'{titles[sub_dir]}')
        axes[axes_ind[sub_dir]].legend(bbox_to_anchor=(
            1.05, 1), loc='upper left', borderaxespad=0., title='Learning Rates')

    # fig.suptitle('Overfitting analysis', fontsize=12)
    # name = f'{len(model_paths[key])}models_{NOTE}.pdf'
    name = f'{key}_loss_final.pdf'
    plt.savefig(
        f'{visuals_dir}/{name}', bbox_inches='tight')
