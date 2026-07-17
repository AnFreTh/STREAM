from typing import Type

import lightning as pl
import torch
import torch.nn as nn


class NeuralBaseModel(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        dataset,
        n_topics=10,
        lr=1e-03,
        lr_patience=15,
        patience=15,
        weight_decay=1e-07,
        lr_factor=0.1,
        batch_size=None,
        **kwargs,
    ):
        super().__init__()

        # Separate dataset from other kwargs
        model_kwargs = {key: value for key, value in kwargs.items() if key != "dataset"}
        self.batch_given = batch_size is not None
        self.batch_size = batch_size

        self.model = model_class(dataset=dataset, n_topics=n_topics, **model_kwargs)
        self.lr = lr
        self.lr_patience = lr_patience
        self.patience = patience
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor

    def training_step(self, batch, batch_idx):

        loss = self.model.compute_loss(batch)
        
        if self.batch_given:
            batch_size = self._get_batch_size(batch)
            self.log(
                "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
            )
        else:
            self.log(
                "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        return loss

    def validation_step(self, batch, batch_idx):

        val_loss = self.model.compute_loss(batch)
        
        if self.batch_given:
            batch_size = self._get_batch_size(batch)
            self.log(
                "val_loss",
                val_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
        else:
            self.log(
                "val_loss",
                val_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return val_loss

    def test_step(self, batch, batch_idx):

        test_loss = self.model.compute_loss(batch)
        
        if self.batch_given:
            batch_size = self._get_batch_size(batch)
            self.log(
                "test_loss",
                test_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
        else:
            self.log(
                "test_loss",
                test_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return test_loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _get_batch_size(self, batch):
        """Infer batch size from batch structure."""
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                elif isinstance(value, list):
                    return len(value)
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        elif isinstance(batch, list):
            return len(batch)
        
        # Fallback
        return 1
