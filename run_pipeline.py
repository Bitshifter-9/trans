import sys, os, json, argparse


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def run_pipeline(input_video, output_dir="output", skip_lipsync=False):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Kannada → Hindi Dubbing Pipeline")
    print("=" * 60)

    print("\n[1/8] Extracting audio...")
    from step1_extract_audio import extract_audio
    s1 = extract_audio(input_video, output_dir)
    audio_wav = s1["audio_path"]

    print("\n[2/8] Transcribing (Kannada → English via Whisper)...")
    from step2_transcribe import transcribe
    transcribe(audio_wav, output_dir)
    s2_meta = os.path.join(output_dir, "step2_meta.json")

    print("\n[2b/8] Cleaning ASR output...")
    from step2b_clean_asr import clean_asr
    clean_asr(s2_meta, output_dir)
    s2b_meta = os.path.join(output_dir, "step2_cleaned.json")

    print("\n[3/8] Translating English → Hindi (NLLB-1.3B)...")
    from step3_translate import translate
    translate(s2b_meta, output_dir)
    s3_meta = os.path.join(output_dir, "step3_meta.json")

    print("\n[4/8] Hindi TTS with voice cloning (F5-TTS)...")
    from step4_tts import synthesize_all
    synthesize_all(s3_meta, audio_wav, output_dir)

    s4_meta = os.path.join(output_dir, "step4_meta.json")

    print("\n[5/8] Matching TTS duration to original timings...")
    from step5_duration_match import match_durations
    match_durations(s4_meta, output_dir)

    s5_meta = os.path.join(output_dir, "step5_meta.json")

    print("\n[6/8] Merging dubbed audio with video...")
    from step6_merge_audio import merge_audio
    merge_audio(s5_meta, input_video, output_dir)

    dubbed_video = os.path.join(output_dir, "dubbed_video.mp4")
    dubbed_audio = os.path.join(output_dir, "dubbed_audio.wav")

    if not skip_lipsync:
        print("\n[7/8] Lip sync (Wav2Lip)...")
        from step7_lipsync import lip_sync
        s7 = lip_sync(input_video, dubbed_audio, output_dir)
        lipsync_video = s7["lipsync_video"]
    else:
        print("\n[7/8] Skipping lip sync.")
        lipsync_video = dubbed_video

    print("\n[8/8] Audio mastering and final encode...")
    from step8_master_encode import master_encode
    s8 = master_encode(lipsync_video, output_dir)

    print("\n" + "=" * 60)
    print(f"  DONE → {s8['final_output']}")
    print("=" * 60)
    return s8["final_output"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kannada to Hindi dubbing pipeline")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--skip-lipsync", action="store_true", help="Skip Wav2Lip lip sync")
    args = parser.parse_args()

    run_pipeline(args.input_video, args.output_dir, args.skip_lipsync)
