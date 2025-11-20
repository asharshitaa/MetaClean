from pathlib import Path
from modules.metadata import clean_metadata

def progress_get(eve: str, data: dict)-> None:
    if eve=="start":
        print(f"\nProcessing file: {Path(data['source']).name}")
    elif eve=="done":
        print("Before Cleaning Score:", data.get("before_score"))
        print(" After Cleaning Score:", data.get("after_score"))
        print(f"Saved cleaned image as: {Path(data['output']).name}")
    elif eve=="error":
        print(f"Error with {data.get('source')}: {data.get('error')}")


def run_cmd(ipaths, out_dir):
    return clean_metadata(ipaths, out_dir, progress_cb=progress_get)


if __name__=="__main__":
    test_images= ["tester.jpg"]
    out_dir= "cleaned_output"
    Path(out_dir).mkdir(exist_ok=True)
    run_cmd(test_images, out_dir)
