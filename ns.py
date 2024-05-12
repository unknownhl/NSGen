import random

import cv2
import numpy as np
import torch

import warnings

from allennlp.modules.elmo import batch_to_ids
from captum.attr import LayerConductance, DeepLift
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import spacy
from src import milannotations, milan
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

nlp = spacy.load('en_core_web_lg')


class LG(object):
    def __init__(self, model, model_name, dataset_name, device, results_dir, name, layer_name):

        super(LG, self).__init__()

        self.device = device
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.results_dir = results_dir
        self.name = name
        self.layer_name = layer_name

        self.pseudoclass_topk = 3 if self.dataset_name == 'ImageNet' else 1

        self.dataset_classes = get_dataset_classes(self.dataset_name)
        self.descriptions = get_descriptions(self.device, self.results_dir)

        self.language_model = embed_select(self.name, self.device)

        self.k = 5
        self.sim = 0.0
        self.current = 0.0

        arg_alpha = {
            'vgg16_bn_ImageNet': 0.72,
            'vgg16_bn_CIFAR10': 0.81,
            'resnet50_ImageNet': 0.72,
            'resnet50_CIFAR10': 0.81,
            'mobilenet_v2_ImageNet': 0.72,
            'mobilenet_v2_CIFAR10': 0.81,
        }

        self.arg = arg_alpha[self.model_name + '_' + self.dataset_name]

        print('参数:', self.arg)

        self.previous_values = []

    def build(self, data_loader):
        self.current = 0.0

    def calculate(self, old_torch_image, new_torch_image, torch_label):
        dataset = TensorDataset(torch.cat([old_torch_image, new_torch_image]), torch.cat([torch_label, torch_label]))
        lens = len(dataset)
        description = []
        for img, label in DataLoader(dataset, batch_size=1):
            des_idx = []
            img, label = img.to(self.device), label.to(self.device)
            img.requires_grad = True
            baselines = torch.zeros_like(img)
            for i, layer in enumerate(self.layer_name):
                for name, module in self.model.named_modules():
                    if name == layer:
                        lc = LayerConductance(self.model, module)
                        hiddens = lc.attribute(img, baselines=baselines, target=label)
                        _, channels, _, _ = hiddens.shape
                        _, idx = hiddens.mean((2, 3)).topk(min(self.k, channels), dim=1, largest=True, sorted=True)
                        for j in range(1):
                            des_idx.append(str(layer) + '.' + str(int(idx[0][j])))
            description.append(','.join(list(set([self.descriptions[x].lower() for x in des_idx]))))

        sample_relabels = self._relabel(self.model, DataLoader(dataset, batch_size=lens))
        entire_des = []
        for i in range(len(sample_relabels)):
            entire_des.append('a photo of a ' + ' or '.join(sample_relabels[i]) + ' with ' + description[i])

        ori_emb = self.language_model.forward(entire_des[0], self.device)
        new_emb = self.language_model.forward(entire_des[1], self.device)
        viz_sim = F.cosine_similarity(ori_emb, new_emb).item()

        if viz_sim < self.arg:
            self.sim = viz_sim
        return self.sim

    def gain(self, sim):
        return sim

    def update(self, sim, gain):
        self.previous_values.append(sim)
        self.current = np.mean(np.array(self.previous_values))
        self.sim = 0.0

    def save(self, path):
        print('Saving recorded %s in %s...' % (path))

    def _relabel(self, model, dataloader, full_label=False):
        was_training = model.training
        _ = model.eval()

        class_embed_collect = {}
        sample_reassign_topk = []

        for i, data_input in enumerate(dataloader):
            with torch.no_grad():
                input = data_input[0]
                out = model(input.to(self.device))
            for idx, label in zip(out, data_input[1].cpu().detach().numpy()):
                if label not in class_embed_collect:
                    class_embed_collect[label] = []
                class_embed_collect[label].append(idx.detach().cpu().numpy())
            sample_reassign_topk.extend(
                np.array(self.dataset_classes)[np.argsort(
                    out.detach().cpu().numpy(), axis=1)[:, -self.pseudoclass_topk:][:, ::-1]].tolist())

        if not full_label:
            sample_reassign_topk = [[x.split(', ')[0] for x in y] for y in sample_reassign_topk]

        if was_training:
            _ = model.train()

        return sample_reassign_topk


def get_dataset_classes(dataset, datapath=''):
    if dataset == 'ImageNet':
        with open(datapath + './classes2synsets/imagenet_synsets.txt', 'r') as f:
            dataset_synsets = f.readlines()
        dataset_splits = [line.split(' ') for line in dataset_synsets]
        key_to_classname = {
            spl[0]: ' '.join(spl[1:]).replace('\n', '')
            for spl in dataset_splits
        }

        with open(datapath + './classes2synsets/imagenet_classes.txt', 'r') as f:
            dataset_classes = f.readlines()
        abstract_dataset_classes = [
            x.strip().replace('\n', '') for x in dataset_classes
        ]
        dataset_classes = [key_to_classname[x] for x in abstract_dataset_classes]

    if dataset == 'CIFAR10':
        with open(datapath + './classes2synsets/cifar10_synsets.txt', 'r') as f:
            dataset_synsets = f.readlines()
        dataset_splits = [line.split(' ') for line in dataset_synsets]
        key_to_classname = {
            spl[0]: ' '.join(spl[1:]).replace('\n', '')
            for spl in dataset_splits
        }

        with open(datapath + './classes2synsets/cifar10_classes.txt', 'r') as f:
            dataset_classes = f.readlines()
        abstract_dataset_classes = [
            x.strip().replace('\n', '') for x in dataset_classes
        ]
        dataset_classes = [key_to_classname[x] for x in abstract_dataset_classes]

    return dataset_classes


def get_descriptions(device, results_dir):
    from src.milannotations import datasets
    dataset = datasets.TopImagesDataset(results_dir)
    decoder = milan.pretrained(milannotations.KEYS.BASE)
    decoder.to(device)
    predictions = decoder.predict(dataset, strategy='rerank', temperature=.2, beam_size=50, device=device)
    descriptions = {}
    for index, description in enumerate(predictions):
        sample = dataset[index]
        descriptions.update({str(sample.layer) + '.' + str(sample.unit): description})
    return descriptions

def embed_select(name, device):
    if name not in ['clip']:
        raise NotImplementedError(
            'Natural Language embedding method {} not available!'.format(name))
    if name == 'clip':
        return ClipLanguageModel(device)


class ClipLanguageModel(torch.nn.Module):
    def __init__(self, device):
        super(ClipLanguageModel, self).__init__()
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

    def forward(self, text, device):
        inputs = self.processor(text=text, images=None, return_tensors='pt', padding=True, truncation=True).to(
            device)
        language_embeds = self.model.get_text_features(**inputs)
        return language_embeds.type(torch.float32)


