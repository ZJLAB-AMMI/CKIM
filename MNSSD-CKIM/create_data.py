from pycocotools.coco import COCO
import os
from tqdm import tqdm
# import skimage.io as io
# import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFile
import PIL
import shutil

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

bboxstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

def mkr(path):
    if not os.path.exists(path):
        print("creating path %s" % path)
        os.makedirs(path)  # 可以创建多级目录

def id2name(coco):
    """
    通过标签的id得到标签的name，因为写xml时需要标签的name,但是annnotation里只有标签的id
    """
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes

def geAllImagesId(coco):
    """
    得到传入数据集里所有图片的id，1.得到标签的种类；2.通过遍历图片种含有该标签来得到图片id
    :param coco: 传入数据集
    :return all_img_ids: list 所有图片的id 
    """
    all_class_ids = [] # 所有class的id，这里有90种
    for ls in coco.dataset['categories']:
        all_class_ids.append(int(ls['id']))
    print("All class ids:")
    print(all_class_ids)
    all_img_ids = []
    for cls_id in all_class_ids:
        img_ids = coco.getImgIds(catIds=cls_id) # 含有这个标签的所有图片
        for img_id in tqdm(img_ids): 
            if img_id not in all_img_ids: # 防止出现重复的id
                all_img_ids.append(img_id)
    print("All img ids got, total: %d" % len(all_img_ids))
    return all_img_ids

def parseAnnotations(anns, filename):
    """
    把coco的ann解析成一个list bboxes,里边存放的是这个图片里的所有bbox信息
    """
    bboxes=[]
    for ann in anns: # 遍历这个图片的所有标注
        class_name = classes[ann['category_id']]  # 从id得到这个标注的名字，
        if 'bbox' in ann:  # 如果标注信息里有bbox
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2] + bbox[0])
            ymax = int(bbox[3] + bbox[1])
            if (ymax==ymin) or (xmax==xmin):
                print("bad annotation!!!!!!!! w or h is zero!")
                print(filename)
                return
            bbox = [class_name, xmin, ymin, xmax, ymax]# 一个目标的信息
            bboxes.append(bbox)# 这个图片里所有目标的信息
    return bboxes

def saveXmlAndImg(coco_dataset_dir, savepath, dataset, filename, bboxes):
    # 将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    # anno
    dst_anno_dir = os.path.join(savepath, dataset, 'Annotations')
    mkr(dst_anno_dir)
    anno_path = dst_anno_dir + '/' + filename[:-3] + 'xml'
    # img
    img_path = coco_dataset_dir + dataset + '/' + filename # 从这里读取图片
    dst_img_dir = os.path.join(savepath, dataset, 'JPEGImages')
    mkr(dst_img_dir)
    dst_imgpath = dst_img_dir + '/' + filename
    img = cv2.imread(img_path)
    shutil.copy(img_path, dst_imgpath)
    # txt
    dst_txt_dir = os.path.join(savepath, dataset)
    mkr(dst_txt_dir)
    dst_txt_path = dst_txt_dir + '/image_idx.txt'
    # write xml
    global headstr
    global tailstr
    global bboxstr
    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    f = open(anno_path, "w")
    f.write(head)
    for bbox in bboxes:
        f.write(bboxstr % (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))
    f.write(tail)
    # write txt
    f = open(dst_txt_path, "a")
    f.write(filename[:-4]+'\n')
    f.close()


if __name__ == '__main__':
    """
    把输入的coco数据集转换成 xml 和 jpg 和所有filename的txt
    """
    home_path = os.environ['HOME']
    coco_dataset_dir = home_path + '/coco_dataset/' # 存放coco数据集的文件夹
    datasets_list = ['train2017']#'train2017','val2017'
    # 转换文件保存路径 
    savepath = home_path + "/COCO/"

    miss = 0
    count = 0
    # 遍历train val test
    for dataset in datasets_list:
        annFile = '{}/annotations/instances_{}.json'.format(coco_dataset_dir, dataset)
        coco = COCO(annFile)
        classes = id2name(coco) # 类别 name -> id
        print("Going to get all image ids. This will take a while...")
        all_img_ids = geAllImagesId(coco)# 所有图片的id
        # counter = 0
        # 遍历所有图片
        for imgId in tqdm(all_img_ids):
            # counter += 1
            # print('image %d'%(counter))
            img = coco.loadImgs(imgId)[0]# 得到img信息
            filename = img['file_name']
            # print(filename)
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None) # 根据图片id得到标注的id
            anns = coco.loadAnns(annIds)  # 得到这个图片的标注信息
            bboxes = parseAnnotations(anns, filename) # 解析这个图片的标注
            if(bboxes):#如果这个图片里有bbox
                saveXmlAndImg(coco_dataset_dir, savepath, dataset, filename, bboxes)
                count += 1
            else:
                print("No bbox found in this img.Skip it!")
                miss += 1
    print("Successfully generate %d images." % count)
    print("Miss %d images." % miss)
    