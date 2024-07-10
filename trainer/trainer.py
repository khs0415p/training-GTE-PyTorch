import torch
import torch.nn as nn

from trainer import BaseTrainer
from utils.train_utils import get_dataloader
from utils import LOGGER, colorstr

class GTETrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        # dataloaders
        self.dataloader = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)


    def icl_loss(self, queries, documents, temperature=0.05):
        if queries.dim() >= 3:
            # mean
            queries = queries.mean(dim=1)
            documents = documents.mean(dim=1)

        batch_size = queries.size(0)

        sim_qd = self.cosine_similarity(queries.unsqueeze(1), documents.unsqueeze(0)) / temperature
        sim_qq = self.cosine_similarity(queries.unsqueeze(1), queries.unsqueeze(0)) / temperature
        sim_dq = self.cosine_similarity(documents.unsqueeze(1), queries.unsqueeze(0)) / temperature
        sim_dd = self.cosine_similarity(documents.unsqueeze(1), documents.unsqueeze(0)) / temperature

        mask = torch.eye(batch_size, device=queries.device).bool()

        z = torch.sum(torch.exp(sim_qd), dim=1) + \
            torch.sum(torch.exp(sim_qq.masked_fill(mask, float('-inf'))), dim=1) + \
            torch.sum(torch.exp(sim_dq), dim=1) + \
            torch.sum(torch.exp(sim_dd.masked_fill(mask, float('-inf'))), dim=1)

        loss = -torch.mean(torch.log(torch.exp(torch.diag(sim_qd)) / z))

        return loss


    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        queries_output = self.model(**model_inputs['queries'])
        documents_output = self.model(**model_inputs['documents'])
        
        queries = queries_output['last_hidden_state']
        documents = documents_output['last_hidden_state']
        loss = self.icl_loss(queries, documents)

        return loss


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        queries_output = self.model(**model_inputs['queries'])
        documents_output = self.model(**model_inputs['documents'])
        
        queries = queries_output['last_hidden_state']
        documents = documents_output['last_hidden_state']
        loss = self.icl_loss(queries, documents)

        return loss

