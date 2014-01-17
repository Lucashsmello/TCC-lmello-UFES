import sys
import os
import re

base_re=re.compile('BASE DE DADOS:\s([^\s]+),\s')
#method_result_re=re.compile('^([^\s;(]+);')


def getMetrics(line,metricsidx):
    return [line[0]]+[line[x] for x in metricsidx]

def getresults(filepath,metrics,methods="ALL"):
    f=open(filepath,'r')
    results=[]
    basename="???"
    metricsidx=[]
    for line in f:
        m=base_re.search(line)
        if(m):
            basename=m.group(1)
        if(line.find('RESULTADOS')>=0):
            for line in f:
                line=line.strip('\n')
                line=line[:-1].split(';')
                for x in metrics:
                    metricsidx+=[[i.find(x) for i in line].index(0)]
                results.append(getMetrics(line,metricsidx))
                break
            break
    for line in f:
        line=line.strip('\n')
        if(len(line.strip())>0):
            line=line[:-1].split(';')
            for i in range(1,len(line)):
                line[i]=float(line[i][:line[i].find('\xc2\xb1')])
            if(methods!="ALL"):
#                if(methods.count(line[0].split(' ')[0].strip())>0):
                if([line[0].split(' ')[0].strip().find(j) for j in methods]>[-1]*len(methods)):
                    results.append(getMetrics(line,metricsidx))
            else:
                results.append(getMetrics(line,metricsidx))

    f.close()

    if(len(results)<2):
        raise Exception(filepath + " HAVE NO RESULTS")

    return [basename,results]


def indexList(l):
    newl=[]
    for r in l:
        nr=[12345]*len(r)
        for i in range(len(r)):
            nr[i]=r.index(i)
        newl.append(nr)
    return newl


def near(x,y):
    return abs(x-y)<=0.001

def posprocessRank(rank,result):
    newr=0
    newranks=[-1]*len(rank)
    eranks_i=[rank.index(0)]
    for i in range(1,len(rank)):
        nr_i=rank.index(i)
        if(not near(result[eranks_i[0]],result[nr_i])):
            if(len(eranks_i)==1):
                newr=rank[eranks_i[0]]
            else:
                newr=(rank[eranks_i[0]]+rank[eranks_i[-1]])/2.0
            for er in eranks_i:
                newranks[er]=newr
            eranks_i=[]
        eranks_i.append(nr_i)

    if(len(eranks_i)==1):
        newr=rank[eranks_i[0]]
    else:
        newr=(rank[eranks_i[0]]+rank[eranks_i[-1]])/2.0
    for er in eranks_i:
        newranks[er]=newr

    return newranks

def makeRank(results):
    allranks=[]
    j=0
    for expR in results:
        ranks=[]
        metrics=expR[0][1:]
        sa_i=metrics.index("Subset Accuracy")
        expR=zip(expR[1:],range(len(expR[1:])))
        for i in range(0,len(metrics)):
            if(metrics[i].find("Loss")<0 and metrics[i].find("Tempo")<0):
                expRsort=sorted(expR,key=lambda k:k[0][i+1],reverse=True)
            else:
                expRsort=sorted(expR,key=lambda k:k[0][i+1])
            ranks.append(zip(*expRsort)[-1])
            
        results[j]=[results[j][0]]+sorted(results[j][1:],key=lambda k:k[sa_i+1])
        newranks=indexList(ranks)

        newranks=zip(*sorted(zip(*newranks),key=lambda k:k[sa_i],reverse=True))
        for i in range(len(newranks)):
            newranks[i]=posprocessRank(newranks[i],zip(*results[j])[i+1][1:])
        allranks.append(newranks)
        j+=1
    
    return allranks

def tocsv(dirpath,  metrics, methods="ALL",makerank=True):
    if(not os.path.exists(dirpath+"/csv")):
        os.mkdir(dirpath+"/csv",0777)
    fout=open(dirpath+"/csv/"+"exps.csv",'w')
    allresults=[]
    basenames=[]
    for f in os.listdir(dirpath):
        if(f.find('exp')>=0):
            if(f.find('expLog')<0):
                if(f.find('~')<0):
                    try:
                        r=getresults(dirpath+f,metrics,methods)
                        basenames.append(r[0])
                        allresults.append(r[1])
                    except Exception as e:
                        print "ERROR in file "+f+" :" + str(e)

  #  print allresults[0]
   # print zip(*allresults[0])
    allranks=[]
    if(makerank):
        allranks=makeRank(allresults)

    print str(len(allresults))+" experimentos"
    maxm=0
    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)
 #       print str(len(r)-1)+ " metodos"
    print str(maxm-1)+ " metodos (MAX)"
    print str(len(allresults[0][0])-1)+ " metricas"
    
#    print str(len(allranks))+"/"+str(len(allresults))
 #   print str(len(allranks[0]))+"/"+str(len(allresults[0]))
  #  print str(len(allranks[0][0]))+"/"+str(len(allresults[0][0]))

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
                        fout.write(str(r[i][j])) #valor da metricas
                        if(makerank and j>0 and i>0):
                            fout.write("("+str(allranks[k][j-1][i-1] + 1)+")")
#                             fout.write("("+str(allranks[k][j-1].index(i-1)+1)+")")
                        fout.write(";")
        fout.write("\n")

    fout.close()


allmetricsnames=["Subset Accuracy","Tempo","Hamming Loss", "Average Precision","Ranking Loss"]

if __name__ == "__main__":
    methods="ALL"
    metrics=allmetricsnames

    if(len(sys.argv)==1):
        print "Usage: " +sys.argv[0]+" DIRPATH [METHODS] [METRICS]" 
        print zip(range(len(allmetricsnames)),allmetricsnames)
    else:
        if(len(sys.argv)>2):
            if(sys.argv[2]=="ALL"):
                methods="ALL"
            else:
                methods=sys.argv[2].split(',')
        if(len(sys.argv)>3):
            idx=sys.argv[3].split(',')
            metrics=[allmetricsnames[int(i)] for i in idx]
            print metrics
            
        
        tocsv(sys.argv[1],metrics,methods)
