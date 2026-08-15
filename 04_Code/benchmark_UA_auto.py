import os
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
from encodec.model import EncodecModel
from encodec.utils import convert_audio
from pesq import pesq, NoUtterancesError, BufferTooShortError
from pystoi import stoi

# --- 參數設定 ---
folder_num = "M07"
data_dir = f"/data/UASpeech/audio/noisereduce/{folder_num}"
output_base_dir = "/data/UASpeech_Reconstructed"
target_sr = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定義你想測試的頻寬清單 (kbps)
# 24kHz 模型支援: 1.5, 3.0, 6.0, 12.0
test_bandwidths = [3.0, 6.0, 12.0, 24.0]

# --- 1. 取得音檔清單 ---
audio_files = []
for folder, _, filenames in os.walk(data_dir):
    for filename in filenames:
        path = os.path.join(folder, filename)
        if filename.endswith(".wav") and not filename.startswith("."):
            audio_files.append(path)


# --- 2. 核心處理函數 ---
def evaluate_bandwidth(bandwidth):
    print(f"\n🚀 啟動測試 - Bandwidth: {bandwidth} kbps")
    
    # 初始化模型
    model = EncodecModel.encodec_model_48khz().to(device)
    model.set_target_bandwidth(bandwidth)
    model.eval()

    # 建立該模型專屬的輸出資料夾 F02_ENCODEC_48_24.0k
    model_save_dir = os.path.join(output_base_dir, f"{folder_num}_ENCODEC_48_{bandwidth}k")
    print(f"重建音訊將儲存在: {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    scores = {"pesq": [], "stoi": []}
    resampler_to_16k = torchaudio.transforms.Resample(model.sample_rate, target_sr)
    

    for path in tqdm(audio_files, desc=f"Processing {bandwidth}k"):
        try:
            # 讀取與基本檢查
            ref_wav, sr = torchaudio.load(path)
            if ref_wav.nelement() == 0 or ref_wav.shape[-1] < 100:
                continue
            
            if ref_wav.dim() == 1: ref_wav = ref_wav.unsqueeze(0)

            # 轉換至模型採樣率
            input_wav = convert_audio(ref_wav, sr, model.sample_rate, model.channels)
            input_wav = input_wav.unsqueeze(0).to(device) # [1, 1, T]

            # Padding 處理 (補齊 stride)
            stride = model.segment_stride if model.segment_stride else 24000
            original_length = input_wav.shape[-1]
            pad_to = (original_length + stride - 1) // stride * stride
            padding_needed = pad_to - original_length
            if padding_needed > 0:
                input_wav = F.pad(input_wav, (0, padding_needed))

            # 推論 (壓縮與重建)
            with torch.no_grad():
                encoded_frames = model.encode(input_wav)
                deg_wav = model.decode(encoded_frames)

            deg_wav = deg_wav.squeeze(0) # [Channels, Time]

            # 儲存重建音訊 (保持模型採樣率 24k)
            relative_path = os.path.relpath(path, data_dir)
            save_path = os.path.join(model_save_dir, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchaudio.save(save_path, deg_wav.cpu(), model.sample_rate)

            # 評估準備 (轉 16k, 單聲道)
            deg_wav_16k = resampler_to_16k(deg_wav.cpu())
            ref_wav_16k = torchaudio.transforms.Resample(sr, target_sr)(ref_wav)

            if deg_wav_16k.shape[0] > 1: deg_wav_16k = deg_wav_16k.mean(0, keepdim=True)
            if ref_wav_16k.shape[0] > 1: ref_wav_16k = ref_wav_16k.mean(0, keepdim=True)

            min_len = min(ref_wav_16k.shape[-1], deg_wav_16k.shape[-1])
            ref_np = ref_wav_16k[:, :min_len].reshape(-1).numpy()
            deg_np = deg_wav_16k[:, :min_len].reshape(-1).numpy()

            # 計算分數
            p_score = pesq(target_sr, ref_np, deg_np, 'wb')
            s_score = stoi(ref_np, deg_np, target_sr, extended=False)
            
            scores["pesq"].append(p_score)
            scores["stoi"].append(s_score)

        except (NoUtterancesError, BufferTooShortError):
            continue
        except Exception as e:
            print(f"檔案錯誤 {path}: {e}")
            continue

    return np.mean(scores["pesq"]), np.mean(scores["stoi"])

# --- 3. 執行循環測試 ---
final_results = []

for bw in test_bandwidths:
    avg_p, avg_s = evaluate_bandwidth(bw)
    final_results.append({
        "bandwidth": bw,
        "pesq": avg_p,
        "stoi": avg_s
    })

# --- 4. 顯示最終對比表 ---
print("\n" + "="*40)
print(f"{'Bandwidth (kbps)':<18} | {'PESQ':<8} | {'STOI':<8}")
print("-" * 40)
for r in final_results:
    print(f"{r['bandwidth']:<18.1f} | {r['pesq']:<8.4f} | {r['stoi']:<8.4f}")
print("="*40)