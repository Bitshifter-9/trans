import subprocess
import json, sys, os


def match_durations(input_meta_path, output_dir="output"):
    with open(input_meta_path) as f:
        tts_data = json.load(f)

    matched_dir = os.path.join(output_dir, "matched_segments")
    os.makedirs(matched_dir, exist_ok=True)

    segments = tts_data["segments"]
    matched_segments = []

    print(f"[Step 5] Matching duration for {len(segments)} segments...")

    for seg in segments:
        idx = seg["index"]
        tts_dur = seg["tts_duration"]
        target_dur = seg["target_duration"]
        wav_in = seg["wav_path"]
        wav_out = os.path.join(matched_dir, f"seg_{idx:04d}.wav")

        if target_dur <= 0:
            continue

        ratio = tts_dur / target_dur
        ratio = max(0.5, min(ratio, 8.0))

        filters = build_tempo_filter(ratio)
        filters += f",apad=whole_dur={target_dur}"

        cmd = [
            "ffmpeg", "-y", "-i", wav_in,
            "-af", filters,
            "-t", str(target_dur),
            "-ar", "24000", "-ac", "1",
            wav_out
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{idx:03d}] ERROR: {result.stderr[:100]}")
            continue

        actual_dur = get_duration(wav_out)
        error = abs(actual_dur - target_dur)

        matched_segments.append({
            "index": idx,
            "start": seg["start"],
            "end": seg["end"],
            "target_duration": target_dur,
            "tts_duration": tts_dur,
            "matched_duration": round(actual_dur, 3),
            "duration_error": round(error, 3),
            "tempo_ratio": round(ratio, 3),
            "hindi": seg["hindi"],
            "wav_path": os.path.abspath(wav_out)
        })

        status = "OK" if error < 0.1 else "WARN"
        print(f"  [{idx:03d}] {tts_dur:.2f}s → {actual_dur:.2f}s (target {target_dur:.2f}s) [{status}]")

    avg_error = sum(s["duration_error"] for s in matched_segments) / max(len(matched_segments), 1)

    info = {
        "total_segments": len(matched_segments),
        "avg_duration_error": round(avg_error, 4),
        "segments": matched_segments
    }

    meta_path = os.path.join(output_dir, "step5_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 5] Duration matching done — avg error: {avg_error:.4f}s")
    return info


def build_tempo_filter(ratio):
    if 0.5 <= ratio <= 2.0:
        return f"atempo={ratio}"

    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r}")
    return ",".join(filters)


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step4_meta.json"
    match_durations(meta)
