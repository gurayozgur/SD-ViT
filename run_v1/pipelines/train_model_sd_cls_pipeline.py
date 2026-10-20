import torch

from models.base import BaseModel
from .base import BasePipeline


class TrainModelSDClsPipeline(BasePipeline):
    """Stochastic shared-classifier deep supervision (Eq. 4 of the paper).

    L = w * L_cls(e_p, y) + (1 - w) * L_cls(e_s, y), where e_p is the full-depth
    embedding, e_s the embedding of a uniformly sampled intermediate block, and both
    are scored by the same margin-softmax classifier (shared prototypes W).
    """

    def __init__(self,
                 model: BaseModel,
                 classifier: BaseModel,
                 optimizer,
                 lr_scheduler,
                 loss_weight: float = 0.7):
        super(TrainModelSDClsPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_weight = loss_weight

    @property
    def module_names_list(self):
        return ['model', 'classifier', 'optimizer', 'lr_scheduler']

    def integrity_check(self, dataset):
        dataset_color_space = dataset.color_space
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_train_transform()

    def make_train_transform(self):
        return self.model.make_train_transform()

    def __call__(self, batch):
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 4:
            inputs, placeholder, targets, thetas = batch
        elif len(batch) == 7:
            inputs, targets, ldmk1, theta1, sample2, ldmk2, theta2 = batch
            if sample2.ndim != 1:
                inputs = torch.cat([inputs, sample2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        else:
            raise ValueError('not supported batch format')

        feats_full, feats_sampled, _ = self.model(inputs)
        targets = targets.to(self.classifier.device)
        loss_full = self.classifier(feats_full, targets)
        loss_sampled = self.classifier(feats_sampled, targets)
        return self.loss_weight * loss_full + (1 - self.loss_weight) * loss_sampled

    def train(self):
        if not self.model.config.freeze:
            self.model.train()
        if not self.classifier.config.freeze:
            self.classifier.train()

    def eval(self):
        self.model.eval()
        self.classifier.eval()
