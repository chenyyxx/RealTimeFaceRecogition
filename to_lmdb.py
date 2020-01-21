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
__author__ = 'Yuxiang Chen'

import lmdb
import os
import common


class to_lmdb:
    def add_embed_to_lmdb(self, id, vector):
        self.db_file = os.path.abspath(common.get_conf('lmdb', 'lmdb_path'))
        id = str(id)
        evn = lmdb.open(self.db_file)
        wfp = evn.begin(write=True)
        wfp.put(key=id.encode(), value=common.embed_to_str(vector).encode())
        wfp.commit()
        evn.close()


if __name__ == '__main__':
    # 插入数据
    embed = to_lmdb()
    embed.add_embed_to_lmdb(12, [1, 2, 0.888333, 0.12343430])

    # 遍历
    evn = lmdb.open(embed.db_file)
    wfp = evn.begin()
    for key, value in wfp.cursor():
        print(key, common.str_to_embed(value))