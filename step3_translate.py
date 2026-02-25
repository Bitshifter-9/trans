from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json, sys, os


def translate(input_meta_path, output_dir="output"):
    with open(input_meta_path) as f:
        asr_data = json.load(f)

    model_name = "facebook/nllb-200-1.3B"
    print(f"[Step 3] Loading NLLB-200 (1.3B): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hindi_token_id = tokenizer.convert_tokens_to_ids("hin_Deva")

    translated_segments = []
    for seg in asr_data["segments"]:
        english_text = seg["text"]
        inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=hindi_token_id,
            max_length=512,
            num_beams=5
        )
        hindi_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "english": english_text,
            "hindi": hindi_text
        })
        print(f"  [{seg['start']:.1f}-{seg['end']:.1f}] {english_text} → {hindi_text}")

    full_hindi = " ".join(s["hindi"] for s in translated_segments)

    info = {
        "model": model_name,
        "source": "en",
        "target": "hi",
        "full_hindi": full_hindi,
        "segments": translated_segments
    }

    meta_path = os.path.join(output_dir, "step3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Step 3] Translation done ({len(translated_segments)} segments)")
    return info


if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "output/step2_meta.json"
    translate(meta)
