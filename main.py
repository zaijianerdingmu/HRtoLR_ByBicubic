import os          #导入os模块主要用于文件的读写
import argparse    #导入argpase主要是用来命令行运行时参数的配置
import cv2         #图像处理模块
 
#parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')    #创建一个参数解析对象
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images", action="store_true")
parser.add_argument('--hr_img_dir', type=str, default=r'C:\Users\17865\Desktop\SRtoDR\input',
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir', type=str, default=r'C:\Users\17865\Desktop\SRtoDR\result',
                    help='path to desired output dir for downsampled images')
args = parser.parse_args()
 
hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir
 
print(args.hr_img_dir)
print(args.lr_img_dir)
 
 
#create LR image dirs
os.makedirs(lr_image_dir + "/X2", exist_ok=True)
os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir + "/X4", exist_ok=True)
os.makedirs(lr_image_dir + "/X6", exist_ok=True)
 
supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")
 
#Downsample HR images
for filename in os.listdir(hr_image_dir):
    if not filename.endswith(supported_img_formats):
        continue
 
    name, ext = os.path.splitext(filename)
 
    #Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])
 
    #Blur with Gaussian kernel of width sigma = 1
    hr_img = cv2.GaussianBlur(hr_img, (0,0), 1, 1)
    #cv2.GaussianBlur(hr_img, (0,0), 1, 1)   其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
    #Downsample image 2x
    lr_image_2x = cv2.resize(hr_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)
 
    cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0]+'x2'+ext), lr_image_2x)
 
    #Downsample image 3x
    lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0]+'x3'+ext), lr_img_3x)
 
    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename.split('.')[0]+'x4'+ext), lr_img_4x)
 
    # Downsample image 6x
    lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1/6, fy=1/6,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0]+'x6'+ext), lr_img_6x)