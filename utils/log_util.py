# -*- coding:utf-8 -*-
# author: Xinge
# @file: log_util.py 


def save_to_log(logdir, logfile, message):
    with open(f'{logdir}/{logfile}', "a") as f:
        f.write(message + '\n')
    return