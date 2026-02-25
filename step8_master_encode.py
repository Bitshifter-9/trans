import subprocess
import json, sys, os


def master_encode(input_video, output_dir="output"):
    output_path = os.path.join(output_dir, "final_output.mp4")

    print(f"[Step 8] Mastering and encoding: {input_video}")

    # Keep only a gentle high-pass and a single-pass loudnorm.
    # Removing acompressor and lowpass prevents pumping artifacts and beep sounds.
    audio_filter = (
        "highpass=f=80,"
        "loudnorm=I=-16:LRA=11:TP=-1.5"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-af", audio_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Step 8] Error: {result.stderr[-500:]}")
        raise RuntimeError("Encoding failed")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Step 8] Final output: {output_path} ({size_mb:.1f} MB)")

    info = {
        "final_output": os.path.abspath(output_path),
        "source_video": os.path.abspath(input_video),
        "size_mb": round(size_mb, 2)
    }

    meta_path = os.path.join(output_dir, "step8_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)

    return info


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "output/lipsync_video.mp4"
    if not os.path.exists(video):
        video = "output/dubbed_video.mp4"
    master_encode(video)
