#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re

base_re=re.compile('BASE DE DADOS:\s([^\s]+),\s')
#method_result_re=re.compile('^([^\s;(]+);')


def getMetrics(line,metricsidx):
    return [line[0]]+[line[x] for x in metricsidx]


def mergeResults(results,metricI):
    methods=list(zip(*results)[0])
    for i in range(len(methods)):
        methods[i]=methods[i].split(' ')[0]
    methods=sorted(set(methods))
    newresults=[]
    for m in methods:
        if(len(m)<1): continue
        newr=[r for r in results if r[0].startswith(m)]

        if(results[0][metricI].find("Loss")<0 and results[0][metricI].find("Tempo")<0 and results[0][metricI].find("Mean Rank")<0):
            newresults.append(max(newr,key=lambda k:k[metricI]))
        else:
            newresults.append(min(newr,key=lambda k:k[metricI]))
    return [results[0]]+newresults
    

def getresults(filepath,metrics,methods="ALL",baseClassifs="ALL",makeMergeResults=False,mergeMetricIndex=1):
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
        if(line.find('NaiveBayes')>=0): continue
        line=line.strip('\n')
        if(len(line.strip())>0):
            line=line[:-1].split(';')
            for i in range(1,len(line)):
                xi=line[i].find('\xc2\xb1')
                if(xi>=0):
                    line[i]=line[i][:xi]
                line[i]=float(line[i])
                if(line[i]<0):
                    line[i]=0
            cangetmethod=True
            if(baseClassifs!="ALL"):
                for basec in baseClassifs:
                    if(line[0].find(basec)>=0):
                        break
                else:
                    cangetmethod=False
            if(cangetmethod==True):                    
                if(methods!="ALL"):
                    if(methods.count(line[0].split(' ')[0].strip())>0):
    #                if([line[0].split(' ')[0].strip().find(j) for j in methods]>[-1]*len(methods)):
                        results.append(getMetrics(line,metricsidx))
                else:
                    results.append(getMetrics(line,metricsidx))

    f.close()

    if(len(results)<2):
        raise Exception(filepath + " HAVE NO RESULTS")

    if(makeMergeResults):
        results=mergeResults(results,results[0].index(allmetricsnames[mergeMetricIndex]))

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

    if(x==y): return True
    return False
    return min(x,y)/max(x,y) >= 0.995
    #return abs(x-y)<=0.001

def posprocessRank(rank,result):
    newr=0
    newranks=[-1]*len(rank)
    eranks_i=[rank.index(0)]
    for i in range(1,len(rank)):
        nr_i=rank.index(i)
        if(not near(result[eranks_i[0]],result[nr_i])):
            newr=(rank[eranks_i[0]]+rank[eranks_i[-1]])/2.0
            for er in eranks_i:
                newranks[er]=newr
            eranks_i=[]
        eranks_i.append(nr_i)

    newr=(rank[eranks_i[0]]+rank[eranks_i[-1]])/2.0
    for er in eranks_i:
        newranks[er]=newr

    return newranks

def makeRank(results,meanRankPesos=[]): #results organizado em metodos
    allranks=[]
    j=0
    makeMeanRank=len(meanRankPesos)>0
    for expR in results:
        ranks=[]
        metrics=expR[0][1:]
        try:
            sa_i=metrics.index("Mean Rank")
        except:
            sa_i=0
            #sa_i=metrics.index("Subset Accuracy")
        expR=zip(expR[1:],range(len(expR[1:])))
        for i in range(0,len(metrics)):
            if(metrics[i].find("Loss")<0 and metrics[i].find("Tempo")<0 and metrics[i].find("Mean Rank")<0 and metrics[i].find("FeatsSub")<0):
                expRsort=sorted(expR,key=lambda k:k[0][i+1],reverse=True)
            else:
                expRsort=sorted(expR,key=lambda k:k[0][i+1])
            ranks.append(zip(*expRsort)[-1])
        
        newranks=indexList(ranks)    #newranks organizado em metricas
        newranksT=zip(*newranks)
        newranksTzipped=zip(newranksT,range(len(newranksT)))
        sorted_idx=zip(*sorted(newranksTzipped,key=lambda k:k[0][sa_i]))[-1]
        
        newranks=zip(*[zip(*newranks)[i] for i in sorted_idx])
        results[j]=[results[j][0]]+[results[j][i+1] for i in sorted_idx]
            
        for i in range(len(newranks)):
            newranks[i]=posprocessRank(newranks[i],zip(*results[j])[i+1][1:])
                
        newranksT=zip(*newranks)

        if(makeMeanRank):
            for res,r in zip(results[j][1:],newranksT):
                meanRank=0.0
                dem=0.0
                for v,p in zip(r,meanRankPesos):
                    meanRank+=float(v)*float(p)
                    dem+=float(p)
                meanRank=meanRank/dem+1
                res.append(meanRank)

            results[j][0].append("Mean Rank")
        
        allranks.append(newranks)
        j+=1
    
    if(makeMeanRank):
        return makeRank(results,[])
    return allranks




def getAllDataMeanRank(allranks,methods,allridx):
    meanrankBases=[0]*len(methods)
    for ranks,ridx in zip(allranks,allridx):
        ranks=ranks[-1]
        meranks=[-1]*len(methods)
        for i in range(len(ridx)):
            meranks[ridx[i]]=ranks[i]
        unkrc=meranks.count(-1)  #unknown rank count
        lastRank=len(methods)-(unkrc-1)/2.0
#        print meranks
        for i in range(len(meranks)):
            if(meranks[i]==-1):
                meranks[i]=lastRank
            meanrankBases[i]+=meranks[i]

    for i in range(len(meanrankBases)):
        meanrankBases[i]=meanrankBases[i]/len(allranks)+1
    sortedFinalResult=sorted(zip(methods,meanrankBases),key=lambda k:k[1])
    methods=zip(*sortedFinalResult)[0]
    meanrankBases=zip(*sortedFinalResult)[1]

    return [methods,meanrankBases]


def getMethods(allresults,makeMergeResults):
    allidx=[]
    methods=[]
    for r in allresults:
        if(len(r)>len(methods)):
            methods=zip(*r)[0][1:]
            if(makeMergeResults):
                methods=[x.split(' ')[0] for x in methods]
    
    for r in allresults:
        rmets=zip(*r)[0][1:] #lista de strings (nome dos metodos)
        if(makeMergeResults):
            rmets=[x.split(' ')[0] for x in rmets]
        ridx=[]
        for rm in rmets:
            try:
                ridx.append(methods.index(rm))
            except:
                pass
        allidx.append(ridx)

    return [methods,allidx]

def saveResults(allresults,fileoutname,basenames,allranks,meanRankAllExps,makeMeanRank,methodsnames,makerank):
    fout=open(fileoutname,'w')
    maxm=0
    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)
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
        
        if(makeMeanRank and i<=len(meanRankAllExps)):
            if(i==0):
                fout.write(" ;"+" ;Media Ranks Todas Bases;")
            else:
                fout.write(" ;"+methodsnames[i-1]+";"+str(meanRankAllExps[i-1])+";")
        fout.write("\n")

    fout.close()
    
def saveResults2(allresults,fileoutname,basenames,allranks,meanRankAllExps,makeMeanRank,methodsnames,makerank):
    fout=open(fileoutname,'w')
    maxm=0

#    print allresults 

    methodsnames=[x.split()[0].replace("-5",'') for x in methodsnames]


    bsortedidx=zip(*sorted(zip(basenames,range(len(basenames)))))[1]
    basenames=sorted(basenames)
    allresults=[allresults[i] for i in bsortedidx]
    allranks=[allranks[i] for i in bsortedidx]

    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)

    for i in range(len(allresults)):
        r=allresults[i]
        idxmets=zip(*sorted(zip(zip(*r)[0],range(len(r)))))[1]
        allresults[i]=[r[x] for x in idxmets]
        ri=allranks[i]
        allranks[i]=zip(*[zip(*ri)[x-1] for x in idxmets[1:]])
        
    
    for i in range(1,len(allresults[0][0])): # para cada metrica
        m=allresults[0][0][i]
        fout.write(m+' ;')
        for j in range(0,maxm): # para cada metodo
            if(j!=0):
                fout.write(str(r[j][0]).split()[0].replace("-5",'')+';') #classificador
            for k in range(0,len(allresults)): # para cada base
                r=allresults[k]
                if(j==0):
                    fout.write(basenames[k].replace('-P','').capitalize()+' ;')
                else:
                    fout.write(str(r[j][i])) #valor da metricas
                    if(makerank and j>0 and i>0):
                        if(int(allranks[k][i-1][j-1])==allranks[k][i-1][j-1]):
                            fout.write("("+str(int(allranks[k][i-1][j-1]) + 1)+")")
                        else:
                            fout.write("("+str(allranks[k][i-1][j-1] + 1)+")")
                    fout.write(";")
            fout.write("\n")
        fout.write("\n")


    for i in range(0,maxm):
        if(makeMeanRank and i<=len(meanRankAllExps)):
            if(i==0):
                fout.write(" ;"+" ;Media dos Ranks de Todas Bases;")
            else:
                fout.write(" ;"+methodsnames[i-1]+";"+str(meanRankAllExps[i-1])+";")
        fout.write("\n")

    fout.close()

def saveResults3(allresults,fileoutname,basenames,allranks,meanRankAllExps,makeMeanRank,methodsnames,makerank):
    fout=open(fileoutname,'w')
    maxm=0

#    print allresults 

    methodsnames=[x.split()[0].replace("-5",'') for x in methodsnames]

    metsidx=zip(*sorted(zip(methodsnames,range(len(methodsnames)))))[1]
    bsortedidx=zip(*sorted(zip(basenames,range(len(basenames)))))[1]
    basenames=sorted(basenames)
    allresults=[allresults[i] for i in bsortedidx]
    allranks=[allranks[i] for i in bsortedidx]
    meanRankAllExps=[meanRankAllExps[i] for i in metsidx]

    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)

    for i in range(len(allresults)):
        r=allresults[i]
        idxmets=zip(*sorted(zip(zip(*r)[0],range(len(r)))))[1]
        allresults[i]=[r[x] for x in idxmets]
        ri=allranks[i]
        allranks[i]=zip(*[zip(*ri)[x-1] for x in idxmets[1:]])
        
    
    for i in range(1,len(allresults[0][0])-1): # para cada metrica
        m=allresults[0][0][i]
        fout.write(m+' ;')
        for j in range(0,maxm): # para cada metodo
            if(j!=0):
                fout.write(str(r[j][0]).split()[0].replace("-5",'').replace("MRLM","RDBR")+';') #classificador
            for k in range(0,len(allresults)): # para cada base
                r=allresults[k]
#                print basenames[k]
                if(j==0):
                    fout.write(basenames[k].replace('-P','').capitalize()+' ;')
                else:
                    fout.write(str(r[j][i])) #valor da metricas
                    if(makerank and j>0 and i>0):
                        if(int(allranks[k][i-1][j-1])==allranks[k][i-1][j-1]):
                            fout.write("("+str(int(allranks[k][i-1][j-1]) + 1)+")")
                        else:
                            fout.write("("+str(allranks[k][i-1][j-1] + 1)+")")
                    fout.write(";")
            if(j>0):
                fout.write(str(meanRankAllExps[j-1])+";")
            else:
                fout.write("Rank médio")
            fout.write("\n")
        fout.write("\n")


    #for i in range(0,maxm):
     #   if(makeMeanRank and i<=len(meanRankAllExps)):
      #      if(i==0):
       #         fout.write(" ;"+" ;Media dos Ranks de Todas Bases;")
        #    else:
         #       fout.write(" ;"+methodsnames[i-1]+";"+str(meanRankAllExps[i-1])+";")
        #   fout.write("\n")

    fout.close()


def tocsv(dirpath, metrics, methods="ALL",baseClassifs="ALL",makerank=True,makeMergeResults=False,mergeMetricIndex=1):
    if(not os.path.exists(dirpath+"/csv")):
        os.mkdir(dirpath+"/csv",0777)
    allresults=[]
    basenames=[]
    metricsNames=[]
    makeMeanRank=False
    if(type(metrics[0])==str):
        metricsNames=metrics
        makeMeanRank=False
    else:
        metricsNames=zip(*metrics)[0]
        makeMeanRank=True

    for f in os.listdir(dirpath):
        if(f.find('exp')>=0):
            if(f.find('expLog')<0):
                if(f.find('~')<0):
                    try:
                        r=getresults(dirpath+f,metricsNames,methods,baseClassifs,makeMergeResults,mergeMetricIndex)
                        basenames.append(r[0])
                        allresults.append(r[1])
                    except Exception as e:
                        print "ERROR in file "+f+" :" + str(e)

    allranks=[]
    if(makerank):
        if(makeMeanRank):
            allranks=makeRank(allresults,zip(*metrics)[1])
            metricsNames=list(metricsNames)
            metricsNames.append("Mean Rank")
        #    newallresults=[]
         #   for res in allresults:
          #      newallresults.append(mergeResults(res))
           # allresults=newallresults
        else:
            allranks=makeRank(allresults)



    print str(len(allresults))+" experimentos"
    maxm=0
    for r in allresults:
        if(len(r)>maxm):
            maxm=len(r)
    print str(maxm-1)+ " metodos (MAX)"
    print str(len(allresults[0][0])-1)+ " metricas"

    meanRankAllExps=[]
    [methodsnames,allridx]=getMethods(allresults,makeMergeResults)
    if(makeMeanRank):
        [methodsnames,meanRankAllExps]=getAllDataMeanRank(allranks,methodsnames,allridx)
    

    saveResults3(allresults,dirpath+"/csv/"+"exps-"+str(baseClassifs)[1:-1]+"-"+str(zip(*metrics)[0])[1:-1]+".csv",basenames,allranks,meanRankAllExps,makeMeanRank,methodsnames,makerank)



allmetricsnames=["Hamming Loss","Subset Accuracy","Example-Based Precision","Example-Based Recall","Example-Based F Measure","Example-Based Accuracy","Example-Based Specificity","Micro-averaged Precision","Micro-averaged Recall","Micro-averaged F-Measure","Micro-averaged Specificity","Macro-averaged Precision","Macro-averaged Recall","Macro-averaged F-Measure","Macro-averaged Specificity","Average Precision","Coverage","OneError","IsError","ErrorSetSize","Ranking Loss","Mean Average Precision","Micro-averaged AUC","Tempo(seg)","Tempo2(seg)","FeatsSub","FeatsSub(perInst)","FeatsSub(perInstFeat)"]

if __name__ == "__main__":
    methods="ALL"
    baseClassifs="ALL"
    metrics=allmetricsnames
    meanRank=[]
    makeMergeResults=False
    mergeMetricIndex=1

    if(len(sys.argv)==1):
        print "Usage: " +sys.argv[0]+" DIRPATH [METHODS] [BASECLASSIFIERS] [METRICS]" 
        print zip(range(len(allmetricsnames)),allmetricsnames)
    else:
        if(len(sys.argv)>2):
            if(sys.argv[2]=="ALL"):
                methods="ALL"
            else:
                methods=sys.argv[2].split(',')
        if(len(sys.argv)>3):
            if(sys.argv[3]=="ALL"):
                baseClassifs="ALL"
            else:
                baseClassifs=sys.argv[3].split(',')
        if(len(sys.argv)>4):
            if(sys.argv[4]=="ALL"):
                metrics=allmetricsnames
            else:

                idx=sys.argv[4].split(',')
                metrics=[allmetricsnames[int(i)] for i in idx]
        for i in range(5,len(sys.argv)):
            if(sys.argv[i].find("-meanrank")>=0):
                pesos=sys.argv[4][len("-meanrank"):]
                pesos=pesos.strip('[')
                pesos=pesos.strip(']')
             #   if(len(pesos)>0):
              #      metrics=zip(metrics,pesos.split(','))
               # else:
                metrics=zip(metrics,[1]*len(metrics))
            if(sys.argv[i].startswith("-merge")):
                makeMergeResults=True
                mergeMetricIndex=int(sys.argv[i][len("-merge"):])
                print mergeMetricIndex

        print metrics
        tocsv(sys.argv[1],metrics,methods,baseClassifs,True,makeMergeResults,mergeMetricIndex)
