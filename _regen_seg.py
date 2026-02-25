"""Regenerate TTS for a single segment index without re-running all of step4."""
import json, sys, os, subprocess
from TTS.api import TTS

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SEG_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUTPUT_DIR = "output"

with open(os.path.join(OUTPUT_DIR, "step3_meta.json")) as f:
    tr_data = json.load(f)

with open(os.path.join(OUTPUT_DIR, "step4_meta.json")) as f:
    meta = json.load(f)

seg = tr_data["segments"][SEG_IDX]
hindi_text = seg["hindi"].strip()
print(f"Regenerating seg{SEG_IDX}: '{hindi_text}'")

tts = TTS(XTTS_MODEL, gpu=False)
ref_clip_path = os.path.join(OUTPUT_DIR, "ref_speaker.wav")

wav_path = os.path.join(OUTPUT_DIR, "tts_segments", f"seg_{SEG_IDX:04d}.wav")
tts.tts_to_file(text=hindi_text, speaker_wav=ref_clip_path, language="hi", file_path=wav_path)

subprocess.run(
    ["ffmpeg", "-y", "-i", wav_path, "-ar", "24000", "-ac", "1", wav_path + ".r.wav"],
    capture_output=True, text=True
)
os.replace(wav_path + ".r.wav", wav_path)

result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", wav_path],
    capture_output=True, text=True
)
duration = float(json.loads(result.stdout)["format"]["duration"])
target = seg["end"] - seg["start"]
print(f"  Generated: {duration:.2f}s (target {target:.1f}s)")

# Patch step4_meta.json for this segment
for s in meta["segments"]:
    if s["index"] == SEG_IDX:
        s["tts_duration"] = round(duration, 3)
        s["hindi"] = hindi_text
        s["wav_path"] = os.path.abspath(wav_path)
        break

with open(os.path.join(OUTPUT_DIR, "step4_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("Done — step4_meta.json updated.")
