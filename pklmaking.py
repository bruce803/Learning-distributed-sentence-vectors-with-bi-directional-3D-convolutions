import numpy as np
import sparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
#import progressbar
try:
    import cPickle as pickle
except ImportError:
    import pickle

def read_from_csv(filePath):
    file = np.genfromtxt(filePath , delimiter="_!_" , dtype=str , invalid_raise = False,encoding='utf-8')
    # print('file',file)
    test_samples = []
    for idx, line in enumerate(file):
        text = ""
        for tx in line[3:]:
            # print('tx',tx)
            text += tx
            text += " "
        # print('text',text)
        label = int(line[1])-100
        if label>=5:
            label=label-1
        if label>=11:
            label=label-1
        test_samples.append((label,text.strip()))
    return test_samples

def eng2img(sentence):
    # print(words_from_sen)
    fontsize = 20
    height = 20
    img_3d = np.zeros((fontsize, height, len(sentence)))

    for i in range(len(sentence)):
        #font = ImageFont.truetype('./fonts/fanti.ttf', fontsize)
        font = ImageFont.truetype('./fonts/simsun.ttc', fontsize)
        # font = ImageFont.truetype('./fonts/FZWangDXCJW.TTF', fontsize)
        #font = ImageFont.truetype('./fonts/SIMHEI.TTF', fontsize)
        mask = font.getmask(sentence[i], mode='L')
        mask_w, mask_h = mask.size
        img = Image.new('L', (height, fontsize), color=0)
        img_w, img_h = img.size
        d = Image.core.draw(img.im, 0)
        d.draw_bitmap(((img_w - mask_w) / 2, (img_h - mask_h) / 2), mask, 255)
        img_map = np.array(img.getdata()).reshape(img.size[1], img.size[0])
        img_3d[..., i] = img_map
        print('imgshape',img_map.shape)
        plt.imshow(img_map, cmap='hot')
    return sparse.COO(img_3d)

def write_pkl(X_list, file_dir):
    print('lenxlist',len(X_list))
    for i in range(len(X_list)):
        arr3d = eng2img(X_list[i][1])
        write_path = file_dir + str(i) + '.pkl'
        print(arr3d)
        print(write_path)
        with open(write_path, 'wb') as curFile:
            pickle.dump([X_list[i][0], arr3d], curFile)

train = "./data/toutiao.txt"
train_samples = read_from_csv(train)
#write_pkl(train_samples, file_dir='./ttpkl/allpkl2/')
write_pkl(train_samples, file_dir='./ttpkl/allpklsimsun/')
#write_pkl(train_samples, file_dir='./ttpkl/allpklHei/')
# write_pkl(train_samples, file_dir='./ttpkl/allpklWangDXCJW/')