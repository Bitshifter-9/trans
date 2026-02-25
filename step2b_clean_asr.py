import re
import json, sys, os


def clean_segments(segments):
    cleaned = []
    seen_texts = set()

    for seg in segments:
        text = seg["text"].strip()

        text = remove_repetitions(text)
        text = fix_common_asr_errors(text)
        text = text.strip()

        if not text or len(text) < 3:
            continue

        if text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())

        cleaned.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })

    return merge_short_segments(cleaned)


def remove_repetitions(text):
    words = text.split()
    if len(words) < 4:
        return text

    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)

    half = len(words) // 2
    first_half = " ".join(words[:half]).lower()
    second_half = " ".join(words[half:]).lower()
    if first_half == second_half and half > 2:
        return " ".join(words[:half])

    return " ".join(result)


def fix_common_asr_errors(text):
    fixes = {
        "bookings": "school",
        "booking": "school",
        "bookshop": "school",
        "book shop": "school",
        "the bookings": "school",
        "the bookshop": "school",
        "ID card": "uniform",
        "nanny": "caretaker",
    }
    for wrong, right in fixes.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if text and text[-1] not in '.?!':
        text += '.'

    return text


def merge_short_segments(segments, min_duration=1.5):
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_dur = prev["end"] - prev["start"]

        if prev_dur < min_duration:
            merged[-1] = {
                "start": prev["start"],
                "end": seg["end"],
                "text": prev["text"].rstrip('.') + ", " + seg["text"][0].lower() + seg["text"][1:]
            }
        else:
            merged.append(seg)

    return merged


def clean_asr(input_meta_path, output_dir="output"):
    with open(input_meta_path) as f:
        asr_data = json.load(f)

    original_count = len(asr_data["segments"])
    cleaned = clean_segments(asr_data["segments"])

    info = {
        "audio_path": asr_data["audio_path"],
        "language": asr_data.get("language", "kn"),
        "output_language": asr_data.get("output_language", "en"),
        "original_segments": original_count,
        "cleaned_segments": len(cleaned),
        "full_text": " ".join(s["text"] for s in cleaned),
        "segments": cleaned
    }

    meta_path = os.path.join(output_dir, "step2_cleaned.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 2b] Cleaned ASR: {original_count} → {len(cleaned)} segments")
    print(f"[Step 2b] Removed {original_count - len(cleaned)} bad/duplicate segments")
    return info


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step2_meta.json"
    clean_asr(meta)
