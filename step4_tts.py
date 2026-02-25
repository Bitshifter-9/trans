import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
import json, sys, os, subprocess


def synthesize_all(input_meta_path, ref_audio_path, output_dir="output"):
    with open(input_meta_path) as f:
        tr_data = json.load(f)

    tts_dir = os.path.join(output_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    print("[Step 4] Loading Chatterbox TTS...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    segments = tr_data["segments"]
    tts_segments = []

    print(f"[Step 4] Generating Hindi speech for {len(segments)} segments...")
    print(f"[Step 4] Using reference voice: {ref_audio_path}")
    print(f"[Step 4] Device: {device}")

    for i, seg in enumerate(segments):
        hindi_text = seg["hindi"]
        if not hindi_text.strip():
            continue

        wav_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")

        wav = model.generate(hindi_text, audio_prompt_path=ref_audio_path)
        torchaudio.save(wav_path, wav, model.sr)

        duration = get_audio_duration(wav_path)
        target_duration = seg["end"] - seg["start"]

        tts_segments.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "target_duration": round(target_duration, 3),
            "tts_duration": round(duration, 3),
            "hindi": hindi_text,
            "wav_path": os.path.abspath(wav_path)
        })

        print(f"  [{i:03d}] {target_duration:.2f}s target | {duration:.2f}s tts | {hindi_text[:40]}")

    info = {
        "tts_engine": "chatterbox",
        "reference_audio": ref_audio_path,
        "device": device,
        "total_segments": len(tts_segments),
        "segments": tts_segments
    }

    meta_path = os.path.join(output_dir, "step4_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 4] TTS done — {len(tts_segments)} audio files generated")
    return info


def get_audio_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step3_meta.json"
    ref = sys.argv[2] if len(sys.argv) > 2 else "output/audio.wav"
    synthesize_all(meta, ref)
