import subprocess
import json, sys, os


WAV2LIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wav2Lip")
CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")


def lip_sync(video_path, audio_path, output_dir="output"):
    output_video = os.path.join(output_dir, "lipsync_video.mp4")

    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)
    output_video_abs = os.path.abspath(output_video)
    print(f"[Step 7] Running Wav2Lip lip sync...")
    print(f"[Step 7] Video: {video_path}")
    print(f"[Step 7] Audio: {audio_path}")

    cmd = [
        sys.executable,
        os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path", CHECKPOINT,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_video_abs,
        "--resize_factor", "1",
        "--nosmooth",
        "--pads", "0", "10", "0", "0",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = WAV2LIP_DIR

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=WAV2LIP_DIR
    )

    if result.returncode != 0:
        print(f"[Step 7] Wav2Lip stderr: {result.stderr[-500:]}")

        if not os.path.exists(output_video_abs):
            print("[Step 7] Wav2Lip failed. Falling back to audio-only dub...")
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                output_video_abs
            ]
            subprocess.run(fallback_cmd, capture_output=True, text=True)
            print(f"[Step 7] Fallback video (no lip sync): {output_video_abs}")

    info = {
        "lipsync_video": output_video_abs,
        "source_video": video_path,
        "source_audio": audio_path,
        "lip_sync_applied": os.path.exists(output_video_abs) and result.returncode == 0
    }

    meta_path = os.path.join(output_dir, "step7_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Step 7] Output: {output_video_abs}")
    return info


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    audio = sys.argv[2] if len(sys.argv) > 2 else "output/dubbed_audio.wav"
    lip_sync(video, audio)
