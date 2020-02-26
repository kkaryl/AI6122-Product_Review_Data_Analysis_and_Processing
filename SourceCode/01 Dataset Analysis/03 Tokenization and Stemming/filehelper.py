# -*- coding: utf-8 -*-
"""
Group 02

AI6122 Text Data Management and Processing
DataSet Analysis Task: Tokenization and Stemming

This FileHelper module contains utilities for file dump writing and loading.

@author: Ong Jia Hui
"""
import os
import pickle

def is_not_empty_file_exists(filepath):
    "Check if file exist and if it is not empty"
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def write_list_to_file(filepath, listdump):
    """Write list into a file using pickle package if filepath is not empty."""
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
        with open(filepath, 'wb') as outfile:
            pickle.dump(listdump, outfile)
        
def load_list_from_file(filepath):
    """Load a file content into list using pickle package."""
    listdump = []
    with open(filepath, 'rb') as infile:
        listdump = pickle.load(infile)
        
    return listdump