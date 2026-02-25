import whisper
import json, sys, os


def transcribe(audio_path, output_dir="output", model_name="small"):
    print(f"[Step 2] Loading Whisper ({model_name})...")
    model = whisper.load_model(model_name)

    print(f"[Step 2] Transcribing Kannada: {audio_path}")
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language="kn",
        verbose=False
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip()
        })

    full_text = " ".join(s["text"] for s in segments)

    info = {
        "audio_path": audio_path,
        "language": "kn",
        "full_text": full_text,
        "segments": segments
    }

    meta_path = os.path.join(output_dir, "step2_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 2] Kannada transcription done ({len(segments)} segments)")
    print(f"[Step 2] Text: {full_text[:200]}...")
    return info


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "output/audio.wav"
    transcribe(audio)
