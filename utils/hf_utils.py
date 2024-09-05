import torch

from utils import TYPE_MODEL
from transformers import AutoModel, TrainingArguments


class CustomDataCollator:
    def __init__(self, padding_id, model_type):
        self.padding_id = padding_id
        self.model_type = model_type
    
    def __call__(self, batch):
        return self.collate_fn(batch, self.padding_id, self.model_type)
        

    def collate_fn(self, batch, padding_value: int = 0, model_type: str = None):
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


def hf_setup(config, tokenizer):
    if config.load_hub:
        model = AutoModel.from_pretrained(config.model_path, cache_dir=config.cache_dir, trust_remote_code=True)
    else:
        model, model_config = TYPE_MODEL[config.model_type]
        model = model(model_config())
    
    training_args = load_hf_args(config)
    data_collator = CustomDataCollator(tokenizer.pad_token_id, config.model_type)

    return model, training_args, data_collator


def load_hf_args(config):
    training_args = TrainingArguments(
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        save_strategy=config.save_strategy,
        learning_rate=config.lr,
        fp16=config.fp16,
        fp16_opt_level= config.fp16_opt_level if config.fp16 else None,
        logging_steps=config.log_step,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        do_eval=config.do_eval,
        lr_scheduler_type=config.scheduler_type,
        max_grad_norm=config.clip_max_norm,
        optim='adamw_hf',
        output_dir='outputs'
    )
    return training_args