import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import BaseTrainer
from utils.train_utils import get_dataloader



class GTETrainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        # dataloaders
        self.dataloader = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def cal_similarity(self, x, y, temperature=0.05):
        sim = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        sim /= temperature

        return sim

    def icl_loss(self, queries, documents, temperature=0.01):
        if queries.dim() >= 3:
            # mean
            queries = queries.mean(dim=1)
            documents = documents.mean(dim=1)

        batch_size = queries.size(0)

        ### test
        # sim_qd = self.cal_similarity(queries, documents)
        # sim_dq = self.cal_similarity(documents, queries)

        # mask = torch.eye(batch_size, device=queries.device).bool()

        # sim_qq = self.cal_similarity(queries, queries)
        # sim_qq = sim_qq.masked_fill(mask, float("-inf"))
        # sim_dd = self.cal_similarity(documents, documents)
        # sim_dd = sim_dd.masked_fill(mask, float("-inf"))


        # labels = torch.arange(batch_size, device=queries.device)

        # qd_loss = self.cross_entropy(sim_qd, labels)
        # dq_loss = self.cross_entropy(sim_dq, labels)
        # qq_loss = self.cross_entropy(sim_qq, labels)
        # dd_loss = self.cross_entropy(sim_dd, labels)

        # loss = (qd_loss + dq_loss + qq_loss + dd_loss) / 4.0
        
        ### origin
        # sim_qd = self.cosine_similarity(queries.unsqueeze(1), documents.unsqueeze(0)) / temperature
        # sim_qq = self.cosine_similarity(queries.unsqueeze(1), queries.unsqueeze(0)) / temperature
        # sim_dq = self.cosine_similarity(documents.unsqueeze(1), queries.unsqueeze(0)) / temperature
        # sim_dd = self.cosine_similarity(documents.unsqueeze(1), documents.unsqueeze(0)) / temperature

        # mask = torch.eye(batch_size, device=queries.device).bool()

        # z = torch.sum(torch.exp(sim_qd), dim=1) + \
        #     torch.sum(torch.exp(sim_qq.masked_fill(mask, float('-inf'))), dim=1) + \
        #     torch.sum(torch.exp(sim_dq), dim=1) + \
        #     torch.sum(torch.exp(sim_dd.masked_fill(mask, float('-inf'))), dim=1)

        # loss = -torch.mean(torch.log(torch.exp(torch.diag(sim_qd)) / z))

        ### modify
        sim_qd = self.cosine_similarity(queries.unsqueeze(1), documents.unsqueeze(0)) / temperature
        sim_qq = self.cosine_similarity(queries.unsqueeze(1), queries.unsqueeze(0)) / temperature
        sim_dq = self.cosine_similarity(documents.unsqueeze(1), queries.unsqueeze(0)) / temperature
        sim_dd = self.cosine_similarity(documents.unsqueeze(1), documents.unsqueeze(0)) / temperature

        mask = torch.eye(batch_size, device=queries.device).bool()

        sim_qq = sim_qq.masked_fill(mask, float("-inf"))
        sim_dd = sim_dd.masked_fill(mask, float("-inf"))
        
        labels = torch.arange(batch_size, device=queries.device)

        qd_loss = self.cross_entropy(sim_qd, labels)
        dq_loss = self.cross_entropy(sim_dq, labels)

        qq_loss = -F.log_softmax(sim_qq, dim=1).masked_select(~mask).mean()
        dd_loss = -F.log_softmax(sim_dd, dim=1).masked_select(~mask).mean()

        loss = (qd_loss + dq_loss + qq_loss + dd_loss) / 4.0

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

        self._backward_step(loss)

        return loss.item()


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

        return loss.item()

