from pathlib import Path
from modules.compress import compress_and_package

#compress rename and zip
def processfile(ipaths, outdir, rprefix="secure"):
    result= compress_and_package(
        ipaths,
        outdir,
        rprefix=rprefix,
        progress_cb=lambda event, data: print(f"{event}: {data}"),
    )
    print(f"ZIP file created at: {result['zip_path']}")
    return result["zip_path"]

#test optional
if __name__=="__main__":
    test_imgs= ["test1.jpg", "test2.png"]  
    output= "compressed_output"
    Path(output).mkdir(exist_ok=True)
    final_zip= processfile(test_imgs, output)
    print("\nFinal ZIP Path:", final_zip)
