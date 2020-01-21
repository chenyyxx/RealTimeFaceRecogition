# -*- coding: utf-8 -*-
"""
 -------------------------------------------------------------------
    File Name: 
    Description: 
    Author: Yuxiang Chen
    Date: 
 -------------------------------------------------------------------
    Change Activity:
    
 -------------------------------------------------------------------
 """
import os
import shutil

main_dir = os.getcwd()
train_dir = main_dir + '/train_img'
os.chdir(train_dir)
name_list = os.listdir()
os.chdir(main_dir)
test_dir = main_dir + '/register_img'
os.chdir(test_dir)

# make directories inside register_img
for d in name_list:
    os.mkdir(d)
os.chdir(main_dir)

# copy files
os.chdir(train_dir)
for d in name_list:
    img_dir = train_dir + '/' + d
    os.chdir(img_dir)
    cp = os.listdir()[0]
    dest = test_dir + '/' + d
    shutil.copy(cp,dest)
