import os
import torch

from typing import List, Dict
from collections import OrderedDict
from transformers import PreTrainedModel, AutoTokenizer
from utils import LOGGER, TYPE_MODEL, colorstr, seed_worker
from utils.data_utils import GTEDataset
from torch.utils.data import DataLoader, random_split, distributed


IGNORE_ID = -100


def get_pretrained_weights(model: PreTrainedModel, pretrained_model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    # shared_layer_names =  set(model.state_dict().keys().intersection(old_model.state_dict().keys()))
    LOGGER.info(colorstr("Change the weights of the model to the base model."))
    layers = []
    for k, v in model.state_dict().items():
        if 'embeddings' in k or 'predictions' in k or 'vocab_projector' in k:
            layers.append((k, v))
            continue
        
        try:
            pretrained_layer = pretrained_model.state_dict()[k]
        except:
            raise ValueError("model and pretrained-model are different.")
        layers.append((k, pretrained_layer))
        
    return OrderedDict(layers)


def collate_fn_warpper(padding_id, model_type):
    def collate_fn_inner(batch):
        return collate_fn(batch, padding_id, model_type)
    return collate_fn_inner


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int = 0, model_type: str = '') -> Dict[str, torch.Tensor]:
    dataset_keys = ['queries', 'documents']
    outputs = {key: None for key in dataset_keys}

    if model_type in ['roberta', 'distilbert']:
        input_keys = ("input_ids",)
    else:
        input_keys = ("input_ids", "token_type_ids")

    for key in dataset_keys:
        temp = {item_key : [instance[key][item_key].squeeze() for instance in batch] for item_key in input_keys}

        # Dynamic padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            temp["input_ids"], batch_first=True, padding_value=padding_value
            )

        attention_mask = input_ids.ne(padding_value).long()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in temp:
            token_type_ids = torch.nn.utils.rnn.pad_sequence(
                temp['token_type_ids'], batch_first=True, padding_value=padding_value
            )
            result['token_type_ids'] = token_type_ids
        
        outputs[key] = result

    return outputs


def get_model(config):
    model, model_config = TYPE_MODEL[config.model_type]
    return model, model_config


def build_dataloader(dataset, batch_size, num_workers, shuffle, pad_token_id, model_type, ddp=False):
    sampler = distributed.DistributedSampler(dataset, shuffle=shuffle) if ddp else None
    return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn_warpper(pad_token_id, model_type),
            worker_init_fn=seed_worker
            )


def get_dataset(config, modes):
    import pickle
    with open(config.data_path, 'rb') as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, cache_dir=config.cache_dir, trust_remote_code=True)
    dataset = GTEDataset(config, data, tokenizer)
    
    if len(modes) > 1:
        train_size = int(len(dataset) * 0.9)
        valid_size = len(dataset) - train_size
        dataset = random_split(dataset, [train_size, valid_size])

    return {mode:ds for mode, ds in zip(modes, dataset)}, tokenizer


def get_dataloader(config):
    """
    Returns:
        (Dict[phase: DataLoader]): dataloader for training
    Examples:
        {'train': DataLoader, 'valid': DataLoader}
    """
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count()
    num_workers = min([4 * n_gpu, config.batch_size // n_gpu, config.batch_size // n_cpu])  # number of workers
    modes = ['train', 'valid'] if config.do_eval else ['train']

    dict_dataset, tokenizer = get_dataset(config, modes)

    dataloader = {mode: build_dataloader(dict_dataset[mode], config.batch_size, num_workers, mode == 'train', tokenizer.pad_token_id, config.model_type,config.ddp) for mode in modes}

    return dataloader