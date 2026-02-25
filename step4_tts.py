import asyncio
import edge_tts
import json, sys, os, subprocess
import numpy as np
import librosa
import soundfile as sf


HINDI_VOICE = "hi-IN-MadhurNeural"


async def generate_tts(text, path):
    await edge_tts.Communicate(text, HINDI_VOICE).save(path)


def get_median_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    valid = f0[~np.isnan(f0)]
    return float(np.median(valid)) if len(valid) > 10 else None


def pitch_shift_to_match(wav_in, ref_pitch_hz, wav_out):
    y, sr = librosa.load(wav_in, sr=None, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    valid = f0[~np.isnan(f0)]
    if len(valid) < 10:
        sf.write(wav_out, y, sr)
        return
    tts_pitch = float(np.median(valid))
    n_steps = 12.0 * np.log2(ref_pitch_hz / tts_pitch)
    n_steps = float(np.clip(n_steps, -8, 8))
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(wav_out, shifted, sr)


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

    print("[Step 4] Analysing original speaker pitch...")
    ref_pitch = get_median_pitch(original_audio_path)
    if ref_pitch:
        print(f"[Step 4] Speaker pitch: {ref_pitch:.1f} Hz")
    else:
        print("[Step 4] Pitch detection failed — skipping pitch match")

    tts_segments = []

    print(f"[Step 4] Generating Hindi TTS + pitch matching for {len(segments)} segments...")

    for i, seg in enumerate(segments):
        hindi_text = seg.get("hindi", "").strip()
        if not hindi_text:
            continue

        mp3_path = os.path.join(tts_dir, f"seg_{i:04d}.mp3")
        raw_wav  = os.path.join(tts_dir, f"seg_{i:04d}_raw.wav")
        wav_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")

        asyncio.run(generate_tts(hindi_text, mp3_path))

        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1", raw_wav],
            capture_output=True, text=True
        )
        os.remove(mp3_path)

        if ref_pitch:
            pitch_shift_to_match(raw_wav, ref_pitch, wav_path)
            os.remove(raw_wav)
        else:
            os.rename(raw_wav, wav_path)

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
        "tts_engine": "edge-tts + pitch-match",
        "voice": HINDI_VOICE,
        "ref_pitch_hz": round(ref_pitch, 2) if ref_pitch else None,
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
