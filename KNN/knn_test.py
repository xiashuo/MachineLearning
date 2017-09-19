# _*_ coding:utf-8 _*_
from numpy import *
def file_matrix(filename):
    fr=open(filename)
    lines=fr.readlines()
    number_lines=len(lines)
    return_matrix=zeros((number_lines,3))
    classlabels=[]
    index=0
    for line in lines:
        line=line.strip()
        listFromLine=line.split('\t')
        return_matrix[index,:]=listFromLine[0:3]
        classlabels.append(listFromLine[-1])
        index+=1
    return return_matrix,classlabels
