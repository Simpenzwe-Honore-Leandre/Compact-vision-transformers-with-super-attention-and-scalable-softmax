
import torch
import os
import pandas as pd
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision,F1Score
from tqdm import tqdm

def train(model,train_loader,val_loader,optimizer,num_classes,save_path,scheduler=None,device='cuda',num_epochs=100):

  # Setup
  acc_metric = Accuracy(task='multiclass', num_classes=num_classes).cuda()
  f1_metric = F1Score(task='multiclass', num_classes=num_classes).cuda()
  top5_acc_metric = Accuracy(task='multiclass', num_classes=num_classes,top_k=5).cuda()

  val_acc_metric = Accuracy(task='multiclass', num_classes=num_classes).cuda()
  val_f1_metric = F1Score(task='multiclass', num_classes=num_classes).cuda()
  val_top5_acc_metric = Accuracy(task='multiclass', num_classes=num_classes,top_k=5).cuda()
  metrics_path = os.path.join( save_path ,"metrics")
  checkpoints_path = os.path.join( save_path ,"checkpoints")
  os.makedirs( checkpoints_path , exist_ok=True)
  os.makedirs( metrics_path, exist_ok=True)

  criterion = nn.CrossEntropyLoss()

  history = []

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      acc_metric.reset()
      top5_acc_metric.reset()
      f1_metric.reset()

      for images, labels in tqdm(train_loader):
          images, labels = images.cuda(), labels.cuda()

          optimizer.zero_grad()

          outputs = model(images)
          loss = criterion(outputs, labels)

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          if scheduler:
            scheduler.step()

          running_loss += loss.item()

          acc_metric.update(outputs, labels)
          top5_acc_metric.update(outputs, labels)
          f1_metric.update(outputs, labels)

      # Compute training metrics
      epoch_loss = running_loss / len(train_loader)

      epoch_acc = acc_metric.compute().item()
      epoch_top5_acc = top5_acc_metric.compute().item()
      epoch_f1 = f1_metric.compute().item()


      # === VALIDATION ===
      model.eval()
      val_loss = 0.0
      val_acc_metric.reset()
      val_top5_acc_metric.reset()
      val_f1_metric.reset()

      with torch.no_grad():
          for val_images, val_labels in val_loader:
              val_images, val_labels = val_images.cuda(), val_labels.cuda()

              val_outputs = model(val_images)
              v_loss = criterion(val_outputs, val_labels)

              val_loss += v_loss.item()
              val_acc_metric.update(val_outputs, val_labels)
              val_f1_metric.update(val_outputs, val_labels)
              val_top5_acc_metric.update(val_outputs, val_labels)


      epoch_val_loss = val_loss / len(val_loader)

      epoch_val_acc = val_acc_metric.compute().item()
      epoch_val_top5_acc = val_top5_acc_metric.compute().item()
      epoch_val_f1 = val_f1_metric.compute().item()


      # Print all metrics
      print(f"Epoch {epoch+1}: "
            f"Train Loss= {epoch_loss:.4e}, Acc= {epoch_acc:.4f}, Top 5 Acc= {epoch_top5_acc:.4f} ,train F1 = {epoch_f1:.4f} | "
            f"Val Loss= {epoch_val_loss:.4e}, Acc= {epoch_val_acc:.4f}, Top 5 val Acc= {epoch_val_top5_acc:.4f}, val F1 = {epoch_val_f1:.4f} ")

      # Save to history
      history.append({
          'epoch': epoch + 1,
          'train_loss': epoch_loss,
          'train_accuracy': epoch_acc,
          'train_top5_accuracy': epoch_top5_acc,
          'train_f1_score': epoch_f1,
          'val_loss': epoch_val_loss,
          'val_accuracy': epoch_val_acc,
          'val_top5_accuracy': epoch_val_top5_acc,
          'val_f1_score': epoch_val_f1
      })



      # Save all metrics to CSV
      # overwrites
      df = pd.DataFrame(history)
      df.to_csv( os.path.join(metrics_path , "transformer_metrics.csv"), index=False,mode='w')
      df.to_csv( os.path.join(metrics_path , "backup_transformer_metrics.csv"), index=False,mode='w')

  return

