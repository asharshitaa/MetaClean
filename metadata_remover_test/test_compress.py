from modules.compress_tools import *

images = ["test1.jpg", "test2.png"]  # make sure these files exist
out = "compressed_output"

print("[1] Starting compression...")
compressed = compress_images(images, out)
print("Compressed files:", compressed)

print("[2] Renaming files...")
new_names = ["secure1.jpg", "secure2.jpg"]
renamed = rename_files(compressed, new_names)
print("Renamed files:", renamed)

print("[3] Creating ZIP...")
zip_file = zip_folder(out)
print("ZIP file:", zip_file)

print("\n✅ Done")
