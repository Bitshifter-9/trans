import subprocess
import json, sys, os


def merge_audio(input_meta_path, video_path, output_dir="output"):
    with open(input_meta_path) as f:
        match_data = json.load(f)

    segments = match_data["segments"]
    merged_wav = os.path.join(output_dir, "dubbed_audio.wav")
    dubbed_video = os.path.join(output_dir, "dubbed_video.mp4")

    video_duration = get_duration(video_path)

    print(f"[Step 6] Merging {len(segments)} segments into full audio track...")
    print(f"[Step 6] Video duration: {video_duration:.2f}s")

    filter_parts = []
    inputs = []

    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg["wav_path"]])
        delay_ms = int(seg["start"] * 1000)
        filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")

    mix_inputs = "".join(f"[d{i}]" for i in range(len(segments)))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(segments)}:duration=longest:normalize=0[mixed]")
    filter_parts.append(f"[mixed]apad=whole_dur={video_duration}[out]")

    filter_str = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    cmd.extend(inputs)
    cmd.extend([
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-ar", "24000", "-ac", "1",
        merged_wav
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Step 6] ERROR merging audio: {result.stderr[-300:]}")
        sys.exit(1)

    print(f"[Step 6] Merged audio: {merged_wav}")

    mux_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", merged_wav,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        dubbed_video
    ]

    result = subprocess.run(mux_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Step 6] ERROR muxing: {result.stderr[-300:]}")
        sys.exit(1)

    info = {
        "merged_audio": os.path.abspath(merged_wav),
        "dubbed_video": os.path.abspath(dubbed_video),
        "video_duration": video_duration,
        "audio_duration": get_duration(merged_wav),
        "total_segments": len(segments)
    }

    meta_path = os.path.join(output_dir, "step6_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Step 6] Dubbed video: {dubbed_video}")
    return info


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step5_meta.json"
    video = sys.argv[2] if len(sys.argv) > 2 else "input.mp4"
    merge_audio(meta, video)
