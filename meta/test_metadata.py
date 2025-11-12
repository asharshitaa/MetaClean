# from modules.metadata_tools import read_exif, remove_exif, compute_privacy_score
# from pathlib import Path

# # Step 1: Point to the existing image in your folder
# image_path = Path("tester.jpg")  # replace with any filename you have

# if not image_path.exists():
#     print(f"❌ File not found: {image_path}")
# else:
#     print(f"🖼️ Processing file: {image_path.name}")

#     # Step 2: Read metadata
#     exif = read_exif(str(image_path))
#     score_before, details_before = compute_privacy_score(exif)
#     print("\nBefore Cleaning:")
#     print("Privacy Score:", score_before)
#     print("Found Metadata:", details_before)

#     # Step 3: Remove metadata
#     cleaned_path = image_path.with_name(f"cleaned_{image_path.name}")
#     remove_exif(str(image_path), str(cleaned_path))

#     # Step 4: Recheck metadata
#     exif_after = read_exif(str(cleaned_path))
#     score_after, details_after = compute_privacy_score(exif_after)
#     print("\nAfter Cleaning:")
#     print("Privacy Score:", score_after)
#     print("Found Metadata:", details_after)

#     print(f"\n✅ Cleaned image saved as: {cleaned_path}")


from pathlib import Path

from modules.metadata import clean_metadata


def _print_progress(event: str, data: dict) -> None:
    if event == "start":
        print(f"\n🖼️ Processing file: {Path(data['source']).name}")
    elif event == "done":
        print("  Before Cleaning Score:", data.get("before_score"))
        print("  After Cleaning Score:", data.get("after_score"))
        print(f"✅ Saved cleaned image as: {Path(data['output']).name}")
    elif event == "error":
        print(f"❌ Error with {data.get('source')}: {data.get('error')}")


def run_cli(image_paths, output_dir):
    return clean_metadata(image_paths, output_dir, progress_cb=_print_progress)


if __name__ == "__main__":
    test_images = ["tester.jpg"]
    output_dir = "cleaned_output"
    Path(output_dir).mkdir(exist_ok=True)
    run_cli(test_images, output_dir)
