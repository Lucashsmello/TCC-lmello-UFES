#!/bin/bash

yy1=(0.25 0.45 0.05 0.50 0.20 0.65 0.85)
yy2=(0.4 0.55 0.15 0.65 0.3 0.75 0.95)
bases=('emotions' 'birds' 'enron' 'scene' 'yeast' 'medical' 'genbase')


for i in {0..6}
do
    b=${bases[$i]}
    gnuplot << EOF
    reset
    set encoding utf8
    set title "${b}"
    set xlabel "Número de iterações"
    set ylabel "Subset Accuracy"
    set yrange [${yy1[$i]}:${yy2[$i]}]
    set terminal postscript
    set output '| ps2pdf - plots/${b}.pdf'
    plot "${b}.data" using 2 title "RDBR" with linespoints ls 7 lt 1 lw 4, \
         "${b}.data" using 3 title "DBR" with linespoints ls 2 lt 1 lw 4
    #pause 2

EOF
done
