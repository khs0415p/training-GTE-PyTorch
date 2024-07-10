import os
import time
import torch
import heapq
import pickle
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import (
    set_seed,
    colorstr,
    TYPE_MODEL,
    LOGGER
)
from utils.file_utils import make_dir, remove_file
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)



class BaseTrainer:
    def __init__(
        self,
        config
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.n_iter = 0

        self.start_epoch = 0
        self.epochs = config.epochs

        self.train_loss_history = []
        self.valid_loss_history = []
        self.learning_rates = []
        self.best_val_loss = float("inf")

        self.save_total_limit = self.config.save_total_limit
        self.saved_path = []

        self.save_path = "{}/{}/{}-{}"
        self.last_save_path = "{}/{}"

        self.phases = ['train', 'valid'] if config.do_eval else ['train']

        set_seed(config.seed)


    def _init_trainer(self):
        # initialize model
        self.init_model()
        # initialize optimizer
        self.init_optimizer(use_exclude=self.config.use_exclude)
        # initialize scheduler
        self.init_scheduler()

        if self.config.fp16:
            from apex import amp
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.config.fp16_opt_level
            )
            if self.config.continuous:
                import pickle
                with open(os.path.join(self.config.checkpoint, 'checkpoint_info.pk', 'rb')) as f:
                    checkpoint_info = pickle.load(f)
                    amp.load_state_dict(checkpoint_info['amp'])
                
                self.learning_rates = checkpoint_info['learining_rates']
                self.best_val_loss = checkpoint_info['best_val_loss']
                self.train_loss_history = checkpoint_info['train_losses']
                self.valid_loss_history = checkpoint_info['valid_losses']
                self.start_epoch = checkpoint_info['epoch_or_step']

        self.model.to(self.device)

        LOGGER.info(f"{colorstr('Initialize'):<25} {colorstr(self.__class__.__name__)}")
        LOGGER.info(f"{colorstr('Batch Size'):<25} {colorstr(str(self.config.batch_size))}")
        LOGGER.info(f"{colorstr('Learning Rate'):<25} {colorstr(str(self.config.lr))}")


    def init_model(self):
        if self.config.load_hub:
            try:
                model = AutoModel.from_pretrained(self.config.model_path, cache_dir=self.config.cache_dir, trust_remote_code=True)
            except:
                raise ValueError(f"Can't load model from hub, Check your hub path ({self.config.model_ath})")
        else:
            model, model_config = TYPE_MODEL[self.config.model_type]
            model = model(model_config())
            if self.config.continuous:
                checkpoint_path = self.config.checkpoint
                try:
                    model_state = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'), map_location=self.device)
                    model = model.load_state_dict(model_state)
                    del model_state
                    torch.cuda.empty_cache()
                except:
                    raise ValueError(f"Not exists file : {checkpoint_path}/pytorch_model.bin")

        self.model = model


    def init_scheduler(self):
        try:
            num_training_steps = len(self.dataloader['train']) * self.config.epochs
        except:
            raise ValueError("Don't exists dataloader")
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError("You have to chioce scheduler type from [linear, cosine].")
        
        if self.config.continuous:
            checkpoint_path = self.config.checkpoint
            try:
                scheduler_state = torch.load(os.path.join(checkpoint_path, 'scheduler.pt'), map_location=self.device)
                scheduler.load_state_dict(scheduler_state)
                del scheduler_state
                torch.cuda.empty_cache()
            except:
                raise ValueError(f"Not exists file : {checkpoint_path}/scheduler.pt")

        self.scheduler = scheduler


    def init_optimizer(self, use_exclude=False):
        optimizer_grouped_parameters = self.model.parameters()
        if use_exclude:
            exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"]
            optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in self.model.named_parameters()
                                if not any(nd in n for nd in exclude_from_weight_decay)
                            ],
                            "weight_decay": self.config.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in self.model.named_parameters()
                                if any(nd in n for nd in exclude_from_weight_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.lr,
        )

        if self.config.continuous:
            checkpoint_path = self.config.checkpoint
            try:
                optimizer_state = torch.load(os.path.join(checkpoint_path, 'optimizer.pt'), map_location=self.device)
                optimizer.load_state_dict(optimizer_state)
                del optimizer_state
                torch.cuda.empty_cache()
            except:
                raise ValueError(f"Not exists file : {checkpoint_path}/optimizer.pt")

        self.optimizer = optimizer


    def _backward_step(self, loss):
        if self.config.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.n_iter % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                clip_grad_norm_(amp.master_params(self.optimizer), self.config.clip_max_norm)
            else:
                clip_grad_norm_(self.model.parameters(), self.config.clip_max_norm)

            self.optimizer.step()
            self.scheduler.step()

        self.n_iter += 1

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            for phase in self.phases:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                epoch_loss = 0
                total_size = 0
                for i, batch in enumerate(tqdm(self.dataloader[phase], total=len(self.dataloader[phase]), desc=f"{phase}|Epoch {epoch+1}")):
                    self.optimizer.zero_grad()

                    batch_size = -1
                    model_inputs = {}
                    for batch_key in batch.keys():
                        if isinstance(batch[batch_key], dict):
                            for key in batch[batch_key].keys():
                                if batch_key not in model_inputs:
                                    model_inputs[batch_key] = {}

                                model_inputs[batch_key][key] = batch[batch_key][key].to(self.device)
                                batch_size = batch[batch_key][key].size(0)
                        else:
                            model_inputs[batch_key] = batch[batch_key].to(self.device)
                            batch_size = batch[batch_key].size(0)

                    step = (epoch * len(self.dataloader[phase])) + i
                    if phase == "train":
                        self._save_learning_rate()
                        loss = self._training_step(model_inputs)

                        self._backward_step(loss)

                        if self.config.save_strategy == "step":
                            if self.n_iter % self.config.save_step == 0:
                                self.save_checkpoint(loss=loss, step=step)
                    else:
                        loss = self._validation_step(model_inputs)

                    loss = loss.item()

                    if i % self.config.log_step == 0:
                        LOGGER.info(f"{colorstr('Epoch'):<25}{colorstr(str(epoch + 1))}")
                        LOGGER.info(f"{colorstr('Step'):<25}{colorstr(str(step))}")
                        LOGGER.info(f"{colorstr('Phase'):<25}{colorstr(phase)}")
                        LOGGER.info(f"{colorstr('Loss'):<25}{colorstr(str(loss))}")
                        if phase == 'train':
                            self.train_loss_history.append([step, loss])
                        else:
                            self.valid_loss_history.append([step, loss])

                    epoch_loss += loss * batch_size
                    total_size += batch_size
                epoch_loss = epoch_loss / total_size

                if self.config.save_strategy == 'epoch':
                    if self.config.do_eval:
                        if phase == 'valid':
                            self.save_checkpoint(epoch_loss, epoch + 1)

                    else:
                        if phase == 'train':
                            self.save_checkpoint(epoch_loss, epoch + 1)
                    
                    LOGGER.info(f"{colorstr('Epoch Loss'):<15}{epoch_loss}")
        self.save_checkpoint(last_save=True)
        LOGGER.info(f"{colorstr('Completed training.')}")


    def _save_checkpoint(
        self,
        base_path: str = None,
        loss: float = None,
        step: int = None,
    ):
        torch.save(self.model.state_dict(), f'{base_path}/pytorch_model.bin')
        torch.save(self.optimizer.state_dict(), f'{base_path}/optimizer.pt')
        torch.save(self.scheduler.state_dict(), f'{base_path}/scheduler.pt')
        self.model.config.to_json_file(f'{base_path}/config.json')

        save_items = {
            'train_losses': self.train_loss_history,
            'valid_losses': self.valid_loss_history,
            'learning_rates': self.learning_rates,
            'best_val_loss': loss,
            'epoch_or_step': step,
        }

        if self.config.fp16:
            from apex import amp
            save_items['amp'] = amp.state_dict()
        
        with open(f'{base_path}/checkpoint-info.pk', 'wb') as f:
            pickle.dump(save_items, f)
        LOGGER.info(f"{colorstr('Saved model...')}")


    def save_checkpoint(
        self,
        loss: float = float("inf"),
        step: int = 0,
        last_save: bool = False
    ):
        cur_time = time.strftime("%m-%d")
        base_path = self.save_path.format(self.config.save_path, cur_time, step, self.config.save_strategy)
        base_path = self.last_save_path.format(self.config.save_path, cur_time) if last_save else base_path
        make_dir(base_path)

        if last_save:
            self._save_checkpoint(
                base_path,
                loss,
                step,
            )
            return

        elif self.best_val_loss > loss or loss == float("inf"):
            if len(self.saved_path) >= self.save_total_limit:
                remove_item = heapq.heappop(self.saved_path)
                remove_file(remove_item[1])
            self._save_checkpoint(base_path, loss, step)
            heapq.heappush(self.saved_path, (-loss, base_path))

            self.best_val_loss = loss


    def _save_learning_rate(self):
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])