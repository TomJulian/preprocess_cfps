## A PIPELINE FOR PROCESSING IMAGES >> FOR CFP ; THIS CODE ASSUMES RGB STYLE IMAGES. 

## The objectives are: (1) reduce the size of images (could be done with PIL too but here we use cv2 as its what automorph had used), (2) if image starts as a png, convert to a jpeg to reduce size (note this introduces loss - but we try minimize), (3) remove 1-channel images (all should be three channel), (4) offer option to flip RE images

## LOAD FUNCTIONS:
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse 

## STEP 1: DEFINE CROP FUNCTIONS. Credit to Automorph and Yukun Zhou for this code, sampled from their Automorph pipeline, available at: https://github.com/rmaphoh/AutoMorph

args = None
root = None
image_new_root = None
FAIL_LOG = None

#  this str2bool fx is used because otherwise you get odd behaviour when specifying true/false flags
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")

def parse_args():
    p = argparse.ArgumentParser(
        description="Crop/compress fundus images; optionally flip RE and/or add filename prefixes."
    )
    p.add_argument("--source_dir", required=True,
                   help="Directory containing images to process.")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write processed images into.")
    p.add_argument("--error_dir", required=True,
                   help="Directory to write failure logs into.")
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="JPEG quality (0-100). Default 95.")
    p.add_argument("--flip_RE", type=str2bool, required=False, default=False,
                   help="true/false. If true, flip right-eye images horizontally.")
    p.add_argument("--add_prefix", type=str2bool, required=False,default=False,
                   help="true/false. If true, add prefix_str to output filenames.")
    p.add_argument("--prefix_str", type=str, default=None,
                   help="Prefix to add when --add_prefix true (e.g. 'FLIP_').")
    p.add_argument("--LE_indicator", type=str, default=None,
                   help="Substring indicating left eye in filenames (required if --flip_RE true).")
    p.add_argument("--RE_indicator", type=str, default=None,
                   help="Substring indicating right eye in filenames (required if --flip_RE true).")
    p.add_argument("--image_resize", type=int, default=1024,
                   help="Resize output to NxN. Default 1024.")
    p.add_argument("--chunksize", type=int, default=250, help="How many images each parallel worker should be passed at once, i defaulted it to 250 assuming we are working with large numbers of images.")
    args = p.parse_args()
    # Cross-argument validation --> this is to stop users submitting commands that don't make sense. 
    if args.add_prefix and not args.prefix_str:
        p.error("--prefix_str is required when --add_prefix true")
    if args.flip_RE:
        if not args.LE_indicator or not args.RE_indicator:
            p.error("--LE_indicator and --RE_indicator are required when --flip_RE true")
        if args.LE_indicator == args.RE_indicator:
            p.error("--LE_indicator and --RE_indicator must be different")
    if not (0 <= args.jpeg_quality <= 100):
        p.error("--jpeg_quality must be between 0 and 100")
    if args.image_resize < 1:
        p.error("--image_resize must be a positive integer")
    return args

def imread(file_path, c=None):
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)
    if im is None:
        raise RuntimeError("Can not read image")
    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imwrite(file_path, image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-5
    #cv2.imshow('gray_img', gray_img)
    #cv2.waitKey()
    #print(threhold)
    _, mask = cv2.threshold(gray_img, max(5,threhold), 1, cv2.THRESH_BINARY)# this converts to a binary - so you have light and dark regions dichotomised
    #cv2.imshow('bz_mask', mask*255)
    #cv2.waitKey()
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    #  cv::floodFill(Temp, Point(0, 0), Scalar(255));
    # _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, [(0, 0),(0,new_mask.shape[0])], (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask


def _get_center_by_edge(mask):
    center=[0,0]
    x=mask.sum(axis=1)
    center[0]=np.where(x>x.max()*0.95)[0].mean()
    x=mask.sum(axis=0)
    center[1]=np.where(x>x.max()*0.95)[0].mean()
    return center


def _get_radius_by_mask_center(mask,center):
    mask=mask.astype(np.uint8)
    ksize=max(mask.shape[1]//400*2+1,3)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # radius=
    index=np.where(mask>0)
    d_int=np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    b_count=np.bincount(np.ceil(d_int).astype(int))
    radius=np.where(b_count>b_count.max()*0.995)[0].max()
    return radius


def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    # center_mask[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=tmp_mask
    # center_mask[bbox[0]:min(bbox[0]+bbox[2],center_mask.shape[0]),bbox[1]:min(bbox[1]+bbox[3],center_mask.shape[1])]=tmp_mask
    return center_mask


def get_mask(img):
    if img.ndim ==3:
        #raise 'image dim is not 3'
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #cv2.imshow('ImageWindow', g_img)
        #cv2.waitKey()
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise ValueError("image dim is not 1 or 3")
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    #g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    center=_get_center_by_edge(tmp_mask)
    #bbox=_get_bbox_by_mask(tmp_mask)
    #print(center)
    #cv2.imshow('ImageWindow', tmp_mask*255)
    #cv2.waitKey()
    radius=_get_radius_by_mask_center(tmp_mask,center)
    #resize back
    #center = [center[0]*2,center[1]*2]
    #radius = int(radius*2)
    center = [center[0], center[1]]
    radius = int(radius)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius


def mask_image(img,mask):
    img[mask<=0,...]=0
    return img


def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border


def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border


def process_without_gb(img):
    # preprocess images
    #   img : origin image
    #   tar_height: height of tar image
    # return:
    #   result_img: preprocessed image
    #   borders: remove border, supplement mask
    #   mask: mask for preprocessed image
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img,bbox=bbox)
    r_img,sup_border = supplemental_black_area(r_img)
    return r_img

## STEP 2: DEFINE NEW FUNCTIONS REQUIRED FOR PROCESSING

def log_fail(p, msg):
    with open(FAIL_LOG, "a") as f:
        f.write(f"{msg}\t{p}\n")


def process_one_image(p):
    name = p.name
    # Decide whether to flip
    do_flip = False
    if args.flip_RE:
        if args.RE_indicator in name:
            do_flip = True
        elif args.LE_indicator in name:
            do_flip = False
        else:
            log_fail(p, "no_eye_indicator_match")
            return
    # Preserve folder structure: mirror input relative path under out_dir
    rel = p.relative_to(root)              # e.g. baseline/subj1/img001.png
    out_path = image_new_root / rel        # e.g. out/baseline/subj1/img001.png
    # Optional prefix on the filename only (not folders!)
    if args.add_prefix:
        out_path = out_path.with_name(f"{args.prefix_str}{out_path.name}")
    # Force JPEG extension - because we are converting to this for reduction of storage space. 
    out_path = out_path.with_suffix(".jpg")
    # Ensure output subfolder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Read the image
    try:
        img = imread(str(p))
    except Exception as e:
        log_fail(p, f"read_fail:{type(e).__name__}")# Log errors/failed files
        return
    # Require RGB -> in early iterations i found sometimes people have converted CFPs to 1 channel images, which is an error
    if img.ndim != 3 or img.shape[2] != 3:
        log_fail(p, "not_rgb")
        return
    # Crop/resize/(optional) flip, then write
    try:
        cropped = process_without_gb(img)
        resized = cv2.resize(
            cropped,
            (args.image_resize, args.image_resize),
            interpolation=cv2.INTER_AREA# I read that this is the best way to downsize. See https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/
        )
        if do_flip:
            resized = cv2.flip(resized, 1)# This is a horizonal flip (i.e. its on the y axis) - 0 would be a vertical flip and would defeat the purpose.
        imwrite(str(out_path), resized)
    except Exception as e:
        log_fail(p, f"crop_fail:{type(e).__name__}:{e}")# Another fail log.
        return

##  STEP 2: POINT TO DIRECTORIES YOU NEED, DEFINE ARGS, MAKE DIRECTORIES IF THEY WERE SPECIFIED BUT DONT EXIST.
def main():
    global args, root, image_new_root, FAIL_LOG
    args = parse_args()
    root = Path(args.source_dir)
    image_new_root = Path(args.out_dir)
    error_root = Path(args.error_dir)
    image_new_root.mkdir(parents=True, exist_ok=True)
    error_root.mkdir(parents=True, exist_ok=True)
    FAIL_LOG = error_root / "failed_images.txt"
    exts = (".jpg", ".jpeg", ".png")
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    n = max(cpu_count() - 1, 1)
    with Pool(processes=n) as pool:
        pool.map(process_one_image, all_files, chunksize=args.chunksize)

if __name__ == "__main__":
    main()

