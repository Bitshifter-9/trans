import subprocess, sys, os, json


def extract_audio(input_video, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.wav")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", input_video],
        capture_output=True, text=True
    )
    probe_data = json.loads(probe.stdout)
    duration = float(probe_data["format"]["duration"])

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_video,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         audio_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    info = {
        "input_video": os.path.abspath(input_video),
        "audio_path": os.path.abspath(audio_path),
        "duration": duration,
        "sample_rate": 16000,
        "channels": 1
    }

    with open(os.path.join(output_dir, "step1_meta.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Step 1] Audio extracted: {audio_path} ({duration:.2f}s)")
    return info


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    extract_audio(video)
