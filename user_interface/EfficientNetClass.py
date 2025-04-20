import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from torch.optim.lr_scheduler import OneCycleLR

from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

# Torchmetrics for evaluation
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    ConfusionMatrix
)

class EfficientNetV2MOptimized(L.LightningModule):
    """
    Enhanced EfficientNetV2-M with Optimized Regularization:
    1. Learning rate scheduling
    2. Progressive unfreezing
    3. Advanced regularization techniques
    4. Mixup data augmentation
    5. Optimized training process
    """

    def __init__(self, n_classes, learning_rate=3e-4, weight_decay=1e-4):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mixup_alpha = 0.2  # Parameter for mixup augmentation

        # Load pre-trained EfficientNetV2-M model
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        
        # IMPORTANT: No need to modify the first conv layer for RGB inputs
        # since the model already expects 3 channels by default
        # REMOVE these lines as they were converting the model to accept grayscale
        # original_conv = self.model.features[0][0]
        # new_conv = nn.Conv2d(...)
        # self.model.features[0][0] = new_conv

        # Initialize frozen layer flags for progressive unfreezing
        self.unfreeze_stage = 0

        # Initially freeze all feature extraction layers
        for i in range(len(self.model.features)):
            for param in self.model.features[i].parameters():
                param.requires_grad = False

        # Get the number of features in the final layer
        in_features = self.model.classifier[1].in_features

        # Replace classifier with enhanced MLP head
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout rate
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),  # SiLU/Swish activation (better than ReLU)
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

        # Use label smoothing cross entropy
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Initialize metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

        # Test metrics
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_precision = Precision(task="multiclass", num_classes=n_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=n_classes, average='macro')
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=n_classes)

        # Save hyperparameters
        self.save_hyperparameters()

    def mixup_data(self, x, y):
        """Apply mixup augmentation to the batch."""
        if self.training and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device)

            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        else:
            return x, y, y, 1.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Apply mixup if training
        if self.training:
            x, y_a, y_b, lam = self.mixup_data(x, y)
            y_pred = self(x)
            loss = lam * self.criterion(y_pred, y_a) + (1 - lam) * self.criterion(y_pred, y_b)
        else:
            y_pred = self(x)
            loss = self.criterion(y_pred, y)

        # Calculate and log metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = self.train_acc(preds, y)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        # Calculate metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = self.val_acc(preds, y)

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        # Calculate predictions
        preds = torch.argmax(y_pred, dim=1)
        probs = torch.softmax(y_pred, dim=1)

        # Update metrics
        self.test_accuracy(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)
        self.test_confusion_matrix(preds, y)

        # Log metrics
        self.log('test_loss', loss, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        # Compute and log final metrics
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1_score = self.test_f1.compute()
        auroc = self.test_auroc.compute()
        conf_mat = self.test_confusion_matrix.compute().cpu().numpy()

        # Store metrics in self for access after training
        self.final_metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1_score.item(),
            'auroc': auroc.item()
        }

        # Print detailed metrics
        print("\n--- Test Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"AUROC: {auroc:.4f}")

        # Reset metrics
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_confusion_matrix.reset()

    def on_train_epoch_start(self):
        """
        Progressive unfreezing of layers based on training epoch.
        """
        # Unfreeze more layers as training progresses
        total_blocks = len(self.model.features)

        if self.current_epoch == 3 and self.unfreeze_stage == 0:
            # Unfreeze the last 30% of feature layers after 3 epochs
            unfreeze_from = int(total_blocks * 0.7)
            for i in range(unfreeze_from, total_blocks):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True
            print(f"Unfreezing layers from {unfreeze_from} to {total_blocks-1}")
            self.unfreeze_stage = 1

        elif self.current_epoch == 6 and self.unfreeze_stage == 1:
            # Unfreeze more layers after 6 epochs
            unfreeze_from = int(total_blocks * 0.4)
            for i in range(unfreeze_from, total_blocks):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True
            print(f"Unfreezing layers from {unfreeze_from} to {total_blocks-1}")
            self.unfreeze_stage = 2

        elif self.current_epoch == 9 and self.unfreeze_stage == 2:
            # Unfreeze all layers after 9 epochs
            for i in range(total_blocks):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True
            print("Unfreezing all layers")
            self.unfreeze_stage = 3

    def configure_optimizers(self):
        """
        Configure optimizer with weight decay and learning rate scheduler.
        """
        # Create optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # OneCycleLR scheduler
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup
                div_factor=25,  # initial_lr = max_lr/25
                final_div_factor=1000,  # min_lr = initial_lr/1000
            ),
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}