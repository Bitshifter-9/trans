import json, sys, os, subprocess
import numpy as np
import soundfile as sf
from f5_tts.api import F5TTS


def extract_reference_clip(source_wav, start, end, out_path, min_dur=5.0, max_dur=12.0):
    duration = min(end - start, max_dur)
    duration = max(duration, min(end - start, min_dur))
    subprocess.run(
        ["ffmpeg", "-y", "-i", source_wav,
         "-ss", str(start), "-t", str(duration),
         "-ar", "24000", "-ac", "1", out_path],
        capture_output=True, text=True
    )


def get_audio_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def synthesize_all(input_meta_path, original_audio_path, output_dir="output"):
    with open(input_meta_path) as f:
        tr_data = json.load(f)

    tts_dir = os.path.join(output_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    segments = tr_data["segments"]

    print("[Step 4] Loading F5-TTS model...")
    tts = F5TTS(device="cpu")

    ref_clip_path = os.path.join(output_dir, "ref_speaker.wav")
    best_ref = max(
        [s for s in segments if (s["end"] - s["start"]) >= 3.0],
        key=lambda s: s["end"] - s["start"],
        default=segments[0]
    )
    extract_reference_clip(
        original_audio_path,
        best_ref["start"],
        best_ref["end"],
        ref_clip_path
    )
    ref_text = best_ref.get("english", "").strip() or "This is a reference audio sample."
    print(f"[Step 4] Reference: {best_ref['start']:.1f}s-{best_ref['end']:.1f}s | text: {ref_text[:60]}")

    tts_segments = []

    print(f"[Step 4] Generating Hindi speech with voice cloning for {len(segments)} segments...")

    for i, seg in enumerate(segments):
        hindi_text = seg.get("hindi", "").strip()
        if not hindi_text:
            continue

        wav_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")

        audio_arr, sr = tts.infer(
            ref_file=ref_clip_path,
            ref_text=ref_text,
            gen_text=hindi_text,
            file_wave=wav_path,
            remove_silence=True,
            seed=42
        )

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

        print(f"  [{i:03d}] {target_duration:.2f}s target | {duration:.2f}s tts | {hindi_text[:50]}")

    info = {
        "tts_engine": "f5-tts",
        "ref_text": ref_text,
        "ref_clip": ref_clip_path,
        "total_segments": len(tts_segments),
        "segments": tts_segments
    }

    meta_path = os.path.join(output_dir, "step4_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 4] TTS done — {len(tts_segments)} audio files generated")
    return info


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step3_meta.json"
    orig_audio = sys.argv[2] if len(sys.argv) > 2 else "output/audio.wav"
    synthesize_all(meta, orig_audio)
