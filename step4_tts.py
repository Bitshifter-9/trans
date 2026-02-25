import json, sys, os, subprocess
from TTS.api import TTS


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def extract_reference_clip(source_wav, segments, out_path, max_dur=10.0):
    best = max(
        [s for s in segments if (s["end"] - s["start"]) >= 3.0],
        key=lambda s: s["end"] - s["start"],
        default=segments[0]
    )
    dur = min(best["end"] - best["start"], max_dur)
    subprocess.run(
        ["ffmpeg", "-y", "-i", source_wav,
         "-ss", str(best["start"]), "-t", str(dur),
         "-ar", "22050", "-ac", "1", out_path],
        capture_output=True, text=True
    )
    return best


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

    print("[Step 4] Loading XTTS-v2 model (voice cloning)...")
    tts = TTS(XTTS_MODEL, gpu=False)

    ref_clip_path = os.path.join(output_dir, "ref_speaker.wav")
    best_ref = extract_reference_clip(original_audio_path, segments, ref_clip_path)
    print(f"[Step 4] Reference clip: {best_ref['start']:.1f}s–{best_ref['end']:.1f}s")

    tts_segments = []

    print(f"[Step 4] Synthesising Hindi with XTTS-v2 voice cloning for {len(segments)} segments...")

    for i, seg in enumerate(segments):
        hindi_text = seg.get("hindi", "").strip()
        if not hindi_text:
            continue

        wav_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")

        tts.tts_to_file(
            text=hindi_text,
            speaker_wav=ref_clip_path,
            language="hi",
            file_path=wav_path
        )

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ar", "24000", "-ac", "1", wav_path + ".r.wav"],
            capture_output=True, text=True
        )
        os.replace(wav_path + ".r.wav", wav_path)

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
        "tts_engine": "xtts-v2",
        "model": XTTS_MODEL,
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
