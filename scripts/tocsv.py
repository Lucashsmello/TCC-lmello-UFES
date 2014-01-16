import sys
import os
import re

base_re=re.compile('BASE DE DADOS:\s([^\s]+),\s')
#method_result_re=re.compile('^([^\s;(]+);')

def getresults(filepath,methods="ALL"):
    f=open(filepath,'r')
    results=[]
    basename="???"

    for line in f:
        m=base_re.search(line)
        if(m):
            basename=m.group(1)
        if(line.find('RESULTADOS')>=0):
            for line in f:
                line=line.strip('\n')
                line=line[:-1].split(';')
                results.append(line)
                break
            break
    for line in f:
        line=line.strip('\n')
        if(len(line.strip())>0):
            line=line[:-1].split(';')
            if(methods!="ALL"):
                if(methods.count(line[0].split(' ')[0].strip())>0):
                    results.append(line)
            else:
                results.append(line)

    f.close()

    return [basename,results]


def tocsv(dirpath, methods="ALL", makerank=True):
    if(not os.path.exists(dirpath+"/csv")):
        os.mkdir(dirpath+"/csv",0777)
    fout=open(dirpath+"/csv/"+"exps.csv",'w')
    allresults=[]
    basenames=[]
    for f in os.listdir(dirpath):
        if(f.find('exp')>=0):
            if(f.find('expLog')<0):
                if(f.find('~')<0):
                    r=getresults(f,methods)
                    basenames.append(r[0])
                    allresults.append(r[1])

    print str(len(allresults))+" experimentos"
    maxm=0
    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)
        print str(len(r)-1)+ " metodos"
    print str(len(allresults[0][0])-1)+ " metricas"
    
    

    for i in range(0,maxm):
        for k in range(0,len(allresults)):
            r=allresults[k]
            
            if(i>=len(r)):
                for j in range(0,len(r[0])):
                    fout.write(" ;")
            else:
                for j in range(0,len(r[i])):
                    if(i==0 and j==0):
                        fout.write(basenames[k]+";")
                    else:
                        fout.write(str(r[i][j])+";")
        fout.write("\n")

    fout.close()


if __name__ == "__main__":
    if(len(sys.argv)>2):
        tocsv(sys.argv[1],sys.argv[2].split(','))
    else:
        tocsv(sys.argv[1])
