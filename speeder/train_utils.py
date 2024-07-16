import lightning as L
from torchmetrics import Accuracy

from torch.nn import CrossEntropyLoss

from speeder.models import *
from speeder.utils import *

class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50(num_classes=200).to(memory_format=torch.channels_last)

        self.loss_fn = CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=200)
        self.val_acc = Accuracy(task="multiclass", num_classes=200)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.log('train_acc', acc, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.log('val_loss', loss, sync_dist=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log('val_acc', acc, sync_dist=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
