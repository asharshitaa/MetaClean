from PIL import Image
import os, zipfile

def compress_images(paths, outfolder, qual=80, maxsize=(1080,1080)):
    os.makedirs(outfolder, exist_ok=True)
    compfile=[]

    for p in paths:
        try:
            im=Image.open(p)
            im.thumbnail(maxsize) #resize
            fname=os.path.basename(p)
            spath= os.path.join(outfolder, fname)

            im.save(spath, optimize=True, quality=qual)
            compfile.append(spath)

        except Exception as e:
            print(f"Compression failed for {p}: {e}")

    return compfile

def rename_files(paths, newnames):
    rfiles=[]
    for path, name in zip(paths, newnames):
        try:
            folder=os.path.dirname(path)
            #if no ext specify, then original
            base, exten=os.path.splitext(name)
            if not exten:
                exten=os.path.splitext(path)[1]
                name=base+exten
            npath=os.path.join(folder, name)

            #no overwrite
            count=1
            fbase, fexten=os.path.splitext(npath)
            while os.path.exists(npath):
                npath=f"{fbase} ({count}){fexten}"
                count+=1
            os.rename(path, npath)
            rfiles.append(npath)

        except Exception as e:
            print(f"Renaming failed for {path}: {e}")
            rfiles.append(path)  #if fail, original

    return rfiles



def zip_folder(path, name="new_zip.zip"):
    zpath=os.path.join(path, name)
    base, exten= os.path.splitext(name)
    count=1
    while os.path.exists(zpath):
        zpath= os.path.join(path, f"{base} ({count}){exten}")
        count+=1
    try:
        with zipfile.ZipFile(zpath, 'w') as zipf:
            for root, x, file in os.walk(path):
                for f in file:
                    if f.lower().endswith(".zip"):
                        continue
                    if f!=name:  #dont zip itself
                        fpath=os.path.join(root, f)
                        zipf.write(fpath, f)

        return zpath
    
    except Exception as e:
        print(f"Zipping failed: {e}")
        return None

