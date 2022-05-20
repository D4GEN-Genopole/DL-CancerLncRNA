from pytorch_lightning import LightningModule
# from pytorch_lightning.metrics import F1, Accuracy, Precision, Recall
# from transformers.optimization import Adafactor
from torch import nn, optim
import wandb



class PytorchModel(LightningModule):
    """Class that implements a pytorch lightning module."""

    def __init__(self, **kwargs):
        super().__init__()
        self.setup_hp(**kwargs)
        self.setup_model(**kwargs)
        self.setup_loss(**kwargs)
        self.setup_metrics(**kwargs)

    def setup_hp(self, lr: float, **kwargs):
        """Setup the hyperparameters."""
        self.lr = lr

    def setup_model(self, model: nn.Module, **kwargs):
        """Setup the model."""
        self.model = model

    def setup_metrics(self, **kwargs):
        """Setup the metrics"""
        pass

    def setup_loss(self, **kwargs):
        """Setup the loss."""
        self.loss= nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Configure the optimizer"""
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def interpret_batch(self, batch):
        x, y = batch
        return x, y

    def forward(self, x,y):
        """Forward."""
        output = self.model(x)
        return self.loss(output, y.reshape(-1, 35))

    def do_training_inference(self, x,y):
        """Inference for training part."""
        loss = self.forward(x,y)
        return loss

    def training_log(self, loss, y_pred, y_true):
        """Do training log."""
        score_wandb = {"Training loss": loss.item()}
        wandb.log(score_wandb)

    def validation_log(self, loss, y_pred, y_true):
        """Do validation log."""
        score_wandb = {"Validation loss": loss.item()}
        return score_wandb

    def training_step(self, batch, batch_nb):
        """Training step."""
        x,y = self.interpret_batch(batch)
        loss = self.do_training_inference(x,y)
        self.training_log(loss, None, None)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        """Training step."""
        x,y  = self.interpret_batch(batch)
        loss = self.do_training_inference(x,y)
        self.validation_log(loss, None, None)
        return {"loss": loss}

    def test_step(self, batch, batch_nb):
        """Training step."""
        x , y = self.interpret_batch(batch)
        loss = self.do_training_inference(x,y)
        return {"loss": loss}
