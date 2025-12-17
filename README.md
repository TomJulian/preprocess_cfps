# overview
This repo is for (1) cropping CFPs (we use code sampled from Automorph for this, credit to: https://github.com/rmaphoh/AutoMorph); (2) optionally flip right eye images if you want to create a dataset where right and left eye images have been converted to the same orientation (can simplify model training), (3) reduce size and convert to JPEG to reduce storage footprint (and you could reduce further than default by reducing image size or jpeg quality). 

The code runs on CPU - there is no need for GPU to run any of the code here. The code is parallelised and will run substantially quicker in multi-CPU set ups. 

# set up
First , download this repo

Then , create the necessary env:
```
conda create -n cfp_preprocessing -c conda-forge python=3.10 opencv -y
conda activate cfp_preprocessing
```
# expected structure
For this to work - the script assumes that all files in the source directory which are image files (jpg, jpeg, png) are CFPs you want to process. The script will attempt to process all images. Therefore, alter the code if that doesnt suit your purposes. 

The outputs should respect the directory structure of the original root (e.g. if your images were split into separate directories for baseline, FU1, FU2 etc this should be respected in the output)

The code discards all 1-channel images, as the expectation is all colour fundus photographs should be three channel . 

# flags
Rhere are several flags you can use to customise the pre-processing:
```
--source_dir # Directory containing images to process.
--out_dir # Directory to write processed images into.
--error_dir # Directory to write failure logs into (a txt file which tells you which images didnt process and approximately why)
--jpeg_quality # JPEG quality (0-100). Default 95. Anything over 95 is probably overkill and will waste storage. You might consider reducing this depending on storage constraints and use case.
--flip_RE # State either true/false. If true, flip right-eye images horizontally. This could simplify model training by reducing image diversity. 
--add_prefix # State either "true/false. If true, add prefix_str to output filenames.
--prefix_str # Prefix to add when --add_prefix true (e.g. 'FLIP_').
--LE_indicator # Substring indicating left eye in filenames (required if --flip_RE true).
--RE_indicator #  Substring indicating right eye in filenames (required if --flip_RE true).
--image_resize # Resize output to NxN. Default 1024.I have made the default fairly high , if you know you are going to resize very small for all your models you might reduce it.
--chunksize" # This is used for parallelisation (which is required to run this quickly); how many images each parallel worker should be passed at once, i defaulted it to 250 assuming we are working with large numbers of images. Note I am not asking how many CPUs you want to use, i have defaulted this to your total CPUs minus 1 (if you're using 1 CPU it will default to 1 rather than trying to allocate 0). 
```
# a example model run
A run might look like: 
```
python /location/you/have/stored/the/preprocessing/script/preprocess_cfps.py \
  --source_dir /where/your/original/images/are/stored/ \
  --out_dir /where/you/want/your/processed/images/to/go/ \
  --error_dir /where/you/want/your/error/log/to/go/ \
  --flip_RE true \
  --LE_indicator "left" \
  --RE_indicator "right" \
  --add_prefix true \
  --prefix_str "FLIP_" \
  --image_resize 1024
```
