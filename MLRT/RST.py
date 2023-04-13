
import cv2
import numpy as np
import glob
import os

def FreCom(img):
    h,w = img.shape[:2]
    img_dct = np.zeros((h,w,3))
    #img_dct = np.fft.fft2(img, axes=(0, 1))
    for i in range(3):
        img_ = img[:, :, i] # 获取rgb通道中的一个
        img_ = np.float32(img_) # 将数值精度调整为32位浮点型
        img_dct[:,:,i] = cv2.dct(img_)  # 使用dct获得img的频域图像

    return img_dct


def idctt(img_f):
    img_out = np.ones_like(img_f)
    for i in range(3):
        img_ = img_f[:, :, i]  # 获取rgb通道中的一个
        img_out[:, :, i] = cv2.idct(img_).clip(0,255)  # 使用dct获得img的频域图像
    return img_out


def Matching(img,reference,alpha=0.2,beta=1):

    #lam = np.random.uniform(alpha, beta)
    seta = np.random.uniform(alpha, beta)

    img_dct=FreCom(img)
    img_l = np.zeros_like(img).astype(np.float64)
    #img_l = img_dct.copy()
    img_l[0,0,:]=img_dct[0,0,:]
    img_dct[0,0,:]=0

    ref_dct=FreCom(reference)
    img_fc = img_dct*seta+ref_dct

    img_out=idctt(img_fc)

    return img_out


if __name__ == '__main__':
    img_path = 'F:\study\code\dataset\\raincityscape_s1\VOC2007\JPEGImages'
    save_path = 'F:\study\code\dataset\\test/'
    image_set_file_train = 'F:\study\code\dataset\watercolor\VOC2007\ImageSets\Main/train.txt'

    img_lists=[]
    with open(image_set_file_train) as f:
        for x in f.readlines():
            if len(x) > 1:
                img_lists.append(img_path+'/'+x.strip()+'.jpg')

    img_basenames = []
    for item in img_lists:
        img_basenames.append(os.path.basename(item))
    i=0
    for img_n, img_p in zip(img_basenames,img_lists):
        img = cv2.imread(img_p)
        print(img_p)
        h1, w1 = img.shape[:2]
        if h1%2!=0 or w1%2!=0:
            img=cv2.resize(img,(w1-w1%2,h1-h1%2),interpolation=cv2.INTER_AREA)

        refrence=np.ones_like(img)
        refrence[:,:,0],refrence[:,:,1],refrence[:,:,2]=refrence[:,:,0]*np.random.randint(0,255),refrence[:,:,1]*np.random.randint(0,255),refrence[:,:,2]*np.random.randint(0,255)
        #refrence *=100
        img_matched = Matching(img,refrence)
        cv2.imwrite(save_path+img_n, img_matched)
        print(i)
        i+=1


