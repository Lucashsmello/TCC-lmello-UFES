/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lmello;

import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import mulan.evaluation.measure.AveragePrecision;

/**
 *
 * @author lmello
 */
public class test {

    public static void main(String[] args) throws Exception {

//        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
//        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String arffFilename = "/home/lmello/mulan-1.4.0/data/emotions.arff";
        String xmlFilename = "/home/lmello/mulan-1.4.0/data/emotions.xml";


        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        IBk knn=new IBk(3);
        MRLM mrlm=new MRLM(knn, 1);
        BinaryRelevance br=new BinaryRelevance(knn);

        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        int numFolds = 5;
        AveragePrecision metric=new AveragePrecision();
        
        results = eval.crossValidate(mrlm, dataset, numFolds);
        System.out.println(results.getMean(metric.getName()));
        results = eval.crossValidate(br, dataset, numFolds);
        System.out.println(results.getMean(metric.getName()));
        

    }
}
