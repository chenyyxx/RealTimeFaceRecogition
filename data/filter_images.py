# -*- coding: utf-8 -*-
"""
 -------------------------------------------------------------------
    File Name: 
    Description: Filter the folder that has less than 4 images
    Author: Yuxiang Chen
    Date: 
 -------------------------------------------------------------------
    Change Activity:
    
 -------------------------------------------------------------------
 """

import os
import shutil
print(os.listdir())
os.chdir(os.getcwd() + '/train_img')
curDir = os.getcwd()
for d in os.listdir():
    sub_dir = curDir+ '/' +d
    count = 0
    os.chdir(sub_dir)
    for s in os.listdir():
        count+=1
    os.chdir(os.pardir)
    print(count)
    print(sub_dir)
    if count<28:
        shutil.rmtree(sub_dir)