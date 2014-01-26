#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import os


def removemm(filename,cDel=';',outfile=""):
    arq=open(filename,'r')
    if(outfile==""):
        outfile=filename+"-NMM"

    arqsaida=open(outfile,'w')

    for line in arq:
        arqsaida.write(line)
        if(line.find("RESULTADOS")>=0):
            break

    for line in arq:
        line=line.replace(';',cDel)
        i=line.find("±")
        if(i>0):
            line_s=line.split(cDel)
            line=""
            for s1 in line_s:
                s1_s=s1.split("±")
                if(len(s1_s)>1):
                    line+=s1_s[0]+cDel
                else:
                    line+=s1+cDel
            
        arqsaida.write(line.replace('\n',"")+'\n')

    arqsaida.close()
    arq.close()



def removemm_DIR(dirpath,cDel=';'):
    if(not os.path.exists(dirpath+"/NMM")):
        os.mkdir(dirpath+"/NMM",0777)

    for f in os.listdir(dirpath):
        if(f.find('exp')>=0 and f.find('NMM')<0):
            if(f.find('expLog')<0):
                if(f.find('~')<0):
                    removemm(dirpath+f,cDel,dirpath+"/NMM/"+f+"-NMM")


if __name__ == "__main__":
    if(len(sys.argv)>1):
        if(os.path.isdir(sys.argv[1])):
            if(len(sys.argv)==3):
                removemm_DIR(sys.argv[1],sys.argv[2])
            else:
                removemm_DIR(sys.argv[1])
        else:
            if(len(sys.argv)==3):
                removemm(sys.argv[1],sys.argv[2])
            else:
                removemm(sys.argv[1])
    else:
        print "Usage:" + sys.argv[0] + " FILE_PATH [TO_CONVERT_DELIMETER=';']"
