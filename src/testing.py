import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import pandas as pd
from final_model import AudioCNN  # same model you used
import os

# ==========================
# 1. Configuration
# ==========================
MODEL_PATH = "FinalModel.pth"
TEST_DIR = "test"   # folder containing .wav files
OUTPUT_CSV = "predictions.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 2. Load model checkpoint
# ==========================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint.get("classes", None)
num_classes = len(classes) if classes else 5  # fallback if missing

model = AudioCNN(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ==========================
# 3. Define preprocessing (same as validation)
# ==========================
transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025,
    ),
    T.AmplitudeToDB()
)

# ==========================
# 4. Helper function
# ==========================
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:  # convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_length = 3 * sample_rate  # pad or truncate to 3s
    if waveform.size(1) < target_length:
        pad_amount = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :target_length]

    spec = transform(waveform)
    max_width = 300
    _, n_mels, time_steps = spec.shape

    if time_steps < max_width:
        pad_amount = max_width - time_steps
        spec = torch.nn.functional.pad(spec, (0, pad_amount))
    elif time_steps > max_width:
        spec = spec[:, :, :max_width]

    return spec

# ==========================
# 5. Predict all files
# ==========================
results = []
test_files = sorted(Path(TEST_DIR).glob("*.wav"))

print(f"Found {len(test_files)} test files.")

with torch.no_grad():
    for file_path in test_files:
        spec = preprocess_audio(file_path)
        spec = spec.unsqueeze(0).to(DEVICE)  # add batch dim

        outputs = model(spec)
        _, predicted = torch.max(outputs, 1)
        pred_class = classes[predicted.item()] if classes else str(predicted.item())

        results.append({
            "Id": file_path.name,
            "Class": pred_class
        })

# ==========================
# 6. Save to CSV
# ==========================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")
