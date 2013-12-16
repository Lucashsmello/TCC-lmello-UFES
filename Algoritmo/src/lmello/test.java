/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lmello;

import java.util.Enumeration;

import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.filters.unsupervised.attribute.Normalize;
import mulan.evaluation.loss.HammingLoss;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.SubsetAccuracy;

/**
 * 
 * @author lmello
 */
public class test {

	public static void main(String[] args) throws Exception {
		String dataDir = "/home/lucasmello/mulan-1.4.0/data/";

		// String arffFilename =
		// "/home/lucasmello/mulan-1.4.0/data/emotions.arff";
		// String xmlFilename =
		// "/home/lucasmello/mulan-1.4.0/data/emotions.xml";
		// String arffFilename =dataDir+"yeast.arff";
		// String xmlFilename = dataDir+"yeast.xml";
		String arffFilename = dataDir + "emotions.arff";
		String xmlFilename = dataDir + "emotions.xml";

		MultiLabelInstances dataset = new MultiLabelInstances(arffFilename,
				xmlFilename);
		 Instances data=dataset.getDataSet();
		 Normalize norm=new Normalize();
		 norm.setInputFormat(data);
		 Instances newdata=norm.getOutputFormat();
		
		 for(int i=0;i<data.numInstances();i++){
		 norm.input(data.instance(i));
		 }
		 norm.batchFinished();
		 Instance processed;
		 while((processed=norm.output())!=null){
		 newdata.add(processed);
		 }
		
		 MultiLabelInstances newMLdataset=new MultiLabelInstances(newdata,
		 dataset.getLabelsMetaData());

		IBk knn = new IBk(7);
		NormalizableDistance nd=(NormalizableDistance)knn.getNearestNeighbourSearchAlgorithm().getDistanceFunction();
//		nd.setDontNormalize(true);
//		knn.setOptions(new String[]{"-I"});
		MRLM mrlm = new MRLM(knn, 7);
		mrlm.setDebug(true);
		BinaryRelevance br = new BinaryRelevance(knn);
		ClassifierChain cc = new ClassifierChain(knn);
		CC2 cc2=new CC2(knn);
		

		Evaluator eval = new Evaluator();
		MultipleEvaluation results;

		int numFolds = 10;
		String[] metrics = { new SubsetAccuracy().getName(),
				new mulan.evaluation.measure.HammingLoss().getName() };

		results = eval.crossValidate(mrlm, dataset, numFolds);
		for (String m : metrics) {
			System.out.println(m + " : " + results.getMean(m));
		}
//		results = eval.crossValidate(br, dataset, numFolds);
//		for (String m : metrics) {
//			System.out.println(m + " : " + results.getMean(m));
//		}
		results = eval.crossValidate(cc, dataset, numFolds);
		for (String m : metrics) {
			System.out.println(m + " : " + results.getMean(m));
		}
//		
//		results = eval.crossValidate(cc2, dataset, numFolds);
//		for (String m : metrics) {
//			System.out.println(m + " : " + results.getMean(m));
//		}

	}
}
