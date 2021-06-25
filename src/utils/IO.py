"""
IO.py
File for handling the data. 

Code by@author: ojgurney-champion
Adapted by@author: dilaratank

Copied and adapted from 
https://github.com/oliverchampion/IVIM-NET/blob/master/IO.py
"""
import os
               
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:
        # print(entry)
        # Create full path, only when ANCHOR
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            if 'other' not in entry and 'oud' not in entry and 'MRE' not in \
                entry and 'output_OGC' not in entry and 'MRs' not in entry:
                    
                allFiles = allFiles + getListOfFiles(fullPath)
            
        elif '.nii.gz' in fullPath:
            if 'IVIM' in entry:
                if os.path.getsize(fullPath)>30000000:
                    temp1,temp2=os.path.split(fullPath)
                    if temp2[0] != 'x':
                        allFiles.append(fullPath)
        elif '.nii' in fullPath:
            if 'IVIM' in entry:
                if os.path.getsize(fullPath)>100000000:
                    temp1,temp2=os.path.split(fullPath)
                    if temp2[0] != 'x':
                        allFiles.append(fullPath)
    return allFiles


def getListOfFiles_matrix(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            if 'DCE' not in entry and '0_' not in entry and 'DIXON' not in entry and 'T1' not in entry and 'dce' not in entry and 'IVIM' not in entry and 'debug' not in entry and 'MASK' not in entry and 'Geen' not in entry and 'Incomplete' not in entry and 'Scripts' not in entry and 'Matlab' not in entry:
                allFiles = allFiles + getListOfFiles_matrix(fullPath)
        elif 'reg_noisy' in fullPath:
            allFiles.append(fullPath)
    return allFiles


def main(dirName,dataset='Anchor'):
    # Get the list of all files in directory tree at given path
    if dataset == 'Anchor':
        listOfFiles = getListOfFiles(dirName)
    elif dataset == 'MATRIX':
        listOfFiles = getListOfFiles_matrix(dirName)
    return listOfFiles        
        
        
        
if __name__ == '__main__':
    main()
