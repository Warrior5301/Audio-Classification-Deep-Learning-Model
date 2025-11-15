from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
# torchaudio.set_audio_backend("sox_io")
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from final_model import AudioCNN
from torch.utils.data import random_split

class CynapticsDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        super().__init__()
        self.data_dir= Path(data_dir)
        self.transform=transform
            
        self.classes=sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx={cls:idx for idx, cls in enumerate(self.classes)}
        
        self.samples=[]
        for cls_name in self.classes:
            class_folder=self.data_dir/cls_name
            for file_path in class_folder.glob("*.wav"):
                self.samples.append((file_path, self.class_to_idx[cls_name]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path,label= self.samples[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0]>1:
            waveform=torch.mean(waveform, dim=0, keepdim=True)
        target_length = 4 * sample_rate  # 4 seconds
        if waveform.size(1) < target_length:
            # Pad at the end with zeros
            pad_amount = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            # Truncate if longer
            waveform = waveform[:, :target_length]
        if self.transform:
            spectrogram=self.transform(waveform)
        else:
            spectrogram=waveform
        max_width = 300  # <-- adjust this if needed
        _, n_mels, time_steps = spectrogram.shape

        if time_steps < max_width:
            pad_amount = max_width - time_steps
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))
        elif time_steps > max_width:
            spectrogram = spectrogram[:, :, :max_width]

        return spectrogram, label

def mixup_data(x,y):
    lam=np.random.beta(0.2,0.2)
    
    batch_size=x.size(0)
    index=torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam*x +(1-lam)*x[index, :]
    y_a,y_b=y,y[index]
    return mixed_x, y_a, y_b,  lam

def mixup_criterion(criterion,pred,y_a,y_b,lam):
    return lam*criterion(pred, y_a)+(1-lam)*criterion(pred,y_b)
    
def train():
    from datetime import datetime
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir=f'./logs{timestamp}'
    writer=SummaryWriter(log_dir)
    data_dir= "train"
    
    train_transform=nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min= 0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )
    val_transform=nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min= 0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )   
    dataset= CynapticsDataset(data_dir="train", transform=train_transform)
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    print(f"Training samples:{train_size}")
    print(f"Val samples:{ val_size}")
    train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
    train_dataloader=DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader=DataLoader(val_dataset,batch_size=32, shuffle=False)
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AudioCNN(num_classes=len(dataset.classes))
    model.to(device)
    
    num_epochs= 200
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=optim.AdamW(model.parameters(),  lr=0.005, weight_decay=0.01)
    
    scheduler= OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )
    
    best_accuracy=0.0
    print("Starting Training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss=0.0
        progress_bar=tqdm(train_dataloader, desc=f'Epoch{epoch+1}/{num_epochs}')
        for data,target in progress_bar:
            data, target=data.to(device), target.to(device)
            
            if np.random.random()>0.7:
                data,target_a,target_b, lam = mixup_data(data,target)
                output=model(data)
                loss=mixup_criterion(criterion,output,target_a,target_b, lam)
            else:
                output=model(data)
                loss=criterion(output,target)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss+=loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        avg_epoch_loss=epoch_loss/len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        #Validation after each epoch
        model.eval()
        
        correct=0
        total=0
        val_loss=0
        
        with torch.no_grad():
            for data,target in test_dataloader:
                data,target=data.to(device), target.to(device)
                outputs=model(data)
                loss=criterion(outputs,target)
                val_loss+=loss.item()  
                
                _, predicted=torch.max(outputs.data, 1)
                total+=target.size(0)
                correct+=(predicted==target).sum().item()
        accuracy=100*correct/total
        avg_val_loss=val_loss/len(test_dataloader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
        print(f'Epoch{epoch+1} Loss:{avg_epoch_loss:.4f}, Val Loss:{avg_val_loss:.4f}, Accuracy:{accuracy:.2f}%')
        
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            torch.save({
                'model_state_dict':model.state_dict(),
                'accuracy':accuracy,
                'epoch':epoch,
                'classes':dataset.classes,
            },'FinalModel.pth')
            print(f'New best model saved:{accuracy:2f}')
    writer.close()
    print(f'Training Completed:Best accuracy:{best_accuracy:2f}')   
if __name__ == "__main__":
    train()
            
            