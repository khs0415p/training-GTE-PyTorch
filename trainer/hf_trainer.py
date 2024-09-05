import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer

class HfTrainer(Trainer):
    def __init__(self, temperature=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def icl_loss(self, queries, documents):
        batch_size = queries.size(0)
        
        if queries.dim() >= 3:
            # mean
            queries = queries.mean(dim=1)
            documents = documents.mean(dim=1)

        sim_qd = self.cosine_similarity(queries.unsqueeze(1), documents.unsqueeze(0)) / self.temperature
        sim_qq = self.cosine_similarity(queries.unsqueeze(1), queries.unsqueeze(0)) / self.temperature
        sim_dq = self.cosine_similarity(documents.unsqueeze(1), queries.unsqueeze(0)) / self.temperature
        sim_dd = self.cosine_similarity(documents.unsqueeze(1), documents.unsqueeze(0)) / self.temperature

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

    def compute_loss(self, model, inputs, return_outputs=False):
        queries_output = self.model(**inputs['queries'])
        documents_output = self.model(**inputs['documents'])

        queries = queries_output['last_hidden_state']
        documents = documents_output['last_hidden_state']

        loss = self.icl_loss(queries, documents)

        return (loss, queries, documents) if return_outputs else loss