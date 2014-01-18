import re
import sys

label_re=re.compile("<label name=\"(.+)\"></label>")
attr_aspaD_re=re.compile(r'@attribute\s+(".*[^\\]")\s.+$')
attr_aspaS_re=re.compile(r"@attribute\s+('.*[^\\]')\s.+$")
attr_re=re.compile("@attribute\s+([^\s]+)\s.+$")

def getLabelsName(fname):
    f=open(fname,'r')
    labels=[]

    for line in f:
        m=label_re.search(line)
        if(m):
            labels+=[m.group(1).strip()]

    f.close()
    return labels

def attrValue(s):
    return int(s[:s.find(' ')])

def transformAttrs(fdata,fdataout,labels):
    labels_written=False
    labels_indexs=[-1]*len(labels)

    attrindex=0
    for line in fdata:
        if(line.find("@data")>=0):
            fdataout.write(line)
            break
        m=attr_aspaS_re.search(line)
        if(not m):
            m=attr_aspaD_re.search(line)
        if(not m):
            m=attr_re.search(line)
        if(m):
            if(not labels_written):
                for l in labels:
                    if(l.strip().count(" ")>0):
                        if(l.count("\\'")>0):
                            fdataout.write('@attribute "'+l.replace('"',r'\"')+'" {0,1}\n')
                        else:
                            fdataout.write("@attribute '"+l.replace("'",r"\'")+"' {0,1}\n")
                    else:
                        fdataout.write("@attribute "+l+" {0,1}\n")
                labels_written=True
            
            try:
                labels_indexs[labels.index(m.group(1).strip().strip("'").replace("\\",""))]=attrindex
            except:
                fdataout.write(line)
            attrindex+=1
        else:
            fdataout.write(line)

    if(not labels_written):
        print "\nERROR 1: labels not written!!!\n"
    if(labels_indexs.count(-1)>0):
        print "\nERROR 2: labels indexs not found:\n"
        x=0
        for l in labels_indexs:
            if(l==-1):
                print "Index of label ("+labels[x]+") not found."
            x+=1

    return labels_indexs

def transformDataValues(fdata,fdataout,labels_indexs):
    linei=1
    for line in fdata:
        print linei
        linei+=1
        if(line.find(',')<0):
            fdataout.write(line)
            continue

        line=line.replace("\n","")
        line=line.replace("\r","")
        attrs=line.split(',')
        if(attrs[0].find(' ')==-1): #se uma base de dados sparso ou nao
            fdataout.write(attrs[labels_indexs[0]])        
            for i in range(1,len(labels_indexs)):
                fdataout.write(','+attrs[labels_indexs[i]])

            for i in range(0,len(attrs)):
                if(labels_indexs.count(i)==0):
                    fdataout.write(','+attrs[i])
            fdataout.write("\n")
        else:
            line=line.strip("{")
            line=line.strip("}")

            for i in range(0,len(labels_indexs)):
                line=line.replace(str(labels_indexs[i])+" ","!@#"+str(i)+' ')

            attrs=line.split(',')
            for i in range(0,len(attrs)):
                if(attrs[i].find("!@#")>=0):
                    attrs[i]=attrs[i][3:attrs[i].find(' ')]+" "+attrs[i].split(' ')[1]
                else:
                    a=attrs[i].split(' ')[0]
                    attrs[i]=str(int(a)+len(labels_indexs))+" "+attrs[i].split(' ')[1]
            
            attrs.sort(key=attrValue)

            for i in range(0,len(attrs)):
                if(i==0):
                    fdataout.write('{'+attrs[i])
                else:
                    fdataout.write(','+attrs[i])
            fdataout.write('}\n')


def transformData(dataPath,labelsPath):
    i=dataPath.rfind('/')+1
    dataoutname=dataPath[:i]+dataPath[i:].replace('.','-P.')
    fdataout=open(dataoutname,'w')
    fdata=open(dataPath,'r')
    labels=getLabelsName(labelsPath)
    
    labels_indexs=transformAttrs(fdata,fdataout,labels)
    print "Writing Data"
    transformDataValues(fdata,fdataout,labels_indexs)
        
    fdataout.close()
    fdata.close()
    return dataoutname

if __name__ == "__main__":
    if(len(sys.argv)>=3):
        transformData(sys.argv[1],sys.argv[2])
    else:
        transformData(sys.argv[1],sys.argv[1].replace(".arff",".xml"))
