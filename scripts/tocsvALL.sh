#!/bin/bash

classifs=('IBk' 'SMO' 'J48' 'Logistic')
metricsids=(0 1 5)


for i in {0..2} # para cada metrica
    do
    m=${metricsids[$i]}
    for j in {0..3} #para cada classificador
    do
        c=${classifs[$j]}
        python tocsv.py ../Algoritmo/exps/expv8/ BR,CC,MRLM-5,DBR,ECC,MCC $c $m -meanrank
    done
done
