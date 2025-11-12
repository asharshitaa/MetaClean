# from modules.compress_tools import *

# images = ["test1.jpg", "test2.png"]  # make sure these files exist
# out = "compressed_output"

# print("[1] Starting compression...")
# compressed = compress_images(images, out)
# print("Compressed files:", compressed)

# print("[2] Renaming files...")
# new_names = ["secure1.jpg", "secure2.jpg"]
# renamed = rename_files(compressed, new_names)
# print("Renamed files:", renamed)

# print("[3] Creating ZIP...")
# zip_file = zip_folder(out)
# print("ZIP file:", zip_file)

# print("\n✅ Done")


from pathlib import Path

from modules.compress import compress_and_package


def process_and_compress(image_paths, output_dir, rename_prefix="secure"):
    """
    Compresses, renames, and zips processed images.
    Returns final zip file path.
    """

    result = compress_and_package(
        image_paths,
        output_dir,
        rename_prefix=rename_prefix,
        progress_cb=lambda event, data: print(f"{event}: {data}"),
    )
    print(f"✅ ZIP file created at: {result['zip_path']}")
    return result["zip_path"]


# Optional standalone test
if __name__ == "__main__":
    test_imgs = ["test1.jpg", "test2.png"]  # make sure they exist
    output = "compressed_output"
    Path(output).mkdir(exist_ok=True)
    final_zip = process_and_compress(test_imgs, output)
    print("\n🎯 Final ZIP Path:", final_zip)
