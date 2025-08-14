import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

class SkinLesionClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')

        self.train_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')

        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = self.train_accuracy(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        val_loss = F.cross_entropy(logits, labels)
        acc = self.val_accuracy(logits, labels)
        self.log("val_loss", val_loss)
        self.log("val_acc", acc)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        test_loss = F.cross_entropy(logits, labels)
        acc = self.test_accuracy(logits, labels)
        precision = self.test_precision(logits, labels)
        recall = self.test_recall(logits, labels)
        f1 = self.test_f1(logits, labels)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        return test_loss

    def configure_optimizers(self):
        classifier_params = self.model.model.classifier.parameters() 
        features_params = self.model.model.features.parameters()  
        optimizer = torch.optim.AdamW([
            {'params': classifier_params, 'lr': self.learning_rate},  
            {'params': features_params, 'lr': 1e-4}, 
        ], weight_decay=1e-4)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True, min_lr=1e-6
        ),
        'monitor': 'val_loss', 
    }
        
        return [optimizer], [scheduler]