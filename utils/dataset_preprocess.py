# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 00:47:10 2018

@author: Administrator
"""
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import matplotlib.pyplot as plt

#def read_jsontolist():
train_json='D:\\bdd100k_labels_images_train.json'
valid_json='D:\\bdd100k_labels_images_val.json'
filer = open (valid_json, 'r', encoding='utf-8')
a= json.load(filer)
filer.close()
labels_classfy=[]
labels_seg=[]
for label in range(len(a)):
    label_detect=[]
    label_seg=[]
    for i in a[label]['labels']:
        if 'box2d' in i.keys():
            label_detect.append(i['category'])
            label_detect.append(i['box2d'])
        elif 'poly2d' in i.keys():
            label_seg.append(i['category'])
            label_seg.append(i['poly2d'])
    labels_classfy.append(label_detect)
    labels_seg.append(label_seg)
    
labels=[]   
for l,ll in enumerate(labels_classfy):
    labels_detect=[]   
    for i in labels_classfy[l]:
        if isinstance(i,str):
            if i=='bike':
                i=0
            elif i=='bus':
                i=1
            elif i=='car':
                i=2
            elif i=='motor':
                i=3
            elif i=='person':
                i=4
            elif i=='rider':
                i=5
            elif i=='traffic light':
                i=6
            elif i=='traffic sign':
                i=7
            elif i=='train':
                i=8
            elif i=='truck':
                i=9
            labels_detect.append(i)
        if isinstance(i,dict):
            for b in i.values():
                value=b
                labels_detect.append(value)
    labels.append(labels_detect)
path='E:\\yolov3\\PyTorch-YOLOv3-master\\data\\labels\\train'    
valid_path='E:\\yolov3\\PyTorch-YOLOv3-master\\data\\labels\\val'
for i,k in enumerate(labels):
    filename = valid_path+"\\"+a[i]['name'].replace('.jpg','.txt')
    #file = open(filename,'w+')
    #file.write(repr(i))
    #file.close()
    np.savetxt(filename,k,'%f')



labels_dir = os.listdir(valid_path)  

image_path=[]
for i in labels_dir:
    list_path='E:/yolov3/PyTorch-YOLOv3-master/data/images/val/'+i
    image_path.append(list_path)               
train_imgpath='image_path.txt'
valid_imgpath='valid_path.txt'
file=open(valid_imgpath,'w+')
for i in image_path:
    file.write(repr(i))
    file.write('\n')
file.close()         


	

#def Kmeans_bbox():
#    #np_labels=np.array(labels).reshape(len(labels),5,-1)
#    box_w=[]
#    box_h=[]
#    box_x=[]
#    box_y=[]
#    for label in labels:
#        for index,cord in enumerate(label):
#            if index%5==1:
#                w=label[index+2]-label[index]
#                x=(label[index+2]+label[index])/2
#                box_w.append(w)
#                box_x.append(x)
#            elif index%5==2:
#                h=label[index+2]-label[index]
#                y=(label[index+2]+label[index])/2
#                box_h.append(h)
#                box_y.append(y)
#        
#        
#        
##    data = np.random.rand(100,2)
#    np_w=np.array(box_w).reshape(np_w.size,1)
#    np_h=np.array(box_h).reshape(np_h.size,1)
#    np_wh=np.concatenate((np_w,np_h),axis=1)
#    estimator=KMeans(n_clusters=9)
#    res=estimator.fit_predict(np_hw)
#    lable_pred=estimator.labels_
#    centroids=estimator.cluster_centers_
#    inertia=estimator.inertia_
#    #print res
#    print(lable_pred)
#    print(centroids)
#    print(inertia)
k-means result£º

43.22139826068694 94.59764940780131
k-means result£º

61.909477790412545 44.85606505359043
k-means result£º

14.791021062274494 15.624941072186187
k-means result£º

34.10552356479446 23.792914785230582
k-means result£º

22.03914811941107 43.29371104408952
k-means result£º

113.52557065283803 78.1342605844463
k-means result£º

430.18247697081256 379.73094961946975
k-means result£º

296.0593961714332 222.8181619206722
k-means result£º

173.53798334594148 146.2725113391473    

w=[15,22,34,43,62,113,173,296,430]
h=[16,43,24,95,45,78,146,223,380]
array_w=np.array(w)
array_h=np.array(h)

new_w=array_w*416/1280
new_h=array_h*416/1280

new_w=[9,13,20,26,37,68,104,178,258]
new_h=[10,26,14,57,27,47,88,134,228]

        
        



           