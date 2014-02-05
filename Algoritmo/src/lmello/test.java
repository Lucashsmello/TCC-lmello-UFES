/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lmello;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import meka.classifiers.multilabel.PCC;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * 
 * @author lmello
 */
public class test {

	public static void main(String[] args) throws Exception {
		String dataDir = "/home/lmello/mulan-1.4.0/data/";

		// String arffFilename =
		// "/home/lucasmello/mulan-1.4.0/data/emotions.arff";
		// String xmlFilename =
		// "/home/lucasmello/mulan-1.4.0/data/emotions.xml";
//		 String arffFilename =dataDir+"yeast.arff";
//		 String xmlFilename = dataDir+"yeast.xml";
		String arffFilename = dataDir + "emotions.arff";
		String xmlFilename = dataDir + "emotions.xml";
		
		MultiLabelInstances dataset = new MultiLabelInstances(arffFilename,
				xmlFilename);
//		 Instances data=dataset.getDataSet();
//		 Normalize norm=new Normalize();
//		 norm.setInputFormat(data);
//		 Instances newdata=norm.getOutputFormat();
//		
//		 for(int i=0;i<data.numInstances();i++){
//		 norm.input(data.instance(i));
//		 }
//		 norm.batchFinished();
//		 Instance processed;
//		 while((processed=norm.output())!=null){
//		 newdata.add(processed);
//		 }
//		
//		 MultiLabelInstances newMLdataset=new MultiLabelInstances(newdata,
//		 dataset.getLabelsMetaData());

		IBk knn = new IBk(11);
//		NormalizableDistance nd=(NormalizableDistance)knn.getNearestNeighbourSearchAlgorithm().getDistanceFunction();
//		nd.setDontNormalize(true);
//		knn.setOptions(new String[]{"-I"});
		knn.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
		MRLM mrlm = new MRLM(knn, 10);
		mrlm.setDebug(true);
		BinaryRelevance br = new BinaryRelevance(knn);
		ClassifierChain cc = new ClassifierChain(knn);
		CC2 cc2=new CC2(knn);
		

		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
		EnsembleOfClassifierChains ecc=new EnsembleOfClassifierChains(knn, 10, true, true);
//		lmelloECC lecc=new lmelloECC(knn, 10, true, false);
		
//		EnsembleMT

		int numFolds = 10;
		String[] metrics = { new SubsetAccuracy().getName(),
				new mulan.evaluation.measure.HammingLoss().getName() };
//
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
		results = eval.crossValidate(ecc, dataset, numFolds);
		for (String m : metrics) {
			System.out.println(m + " : " + results.getMean(m));
		}
////		
//		results = eval.crossValidate(lecc, dataset, numFolds);
//		for (String m : metrics) {
//			System.out.println(m + " : " + results.getMean(m));
//		}

	}
	
//	public static void main(String[] args) throws Exception {
//		long t1=System.nanoTime();
//		superPowerLucasExperiment();
//		System.out.println("tempo1="+(System.nanoTime()-t1)/1e9);
////		System.out.println("$$$$$$$$$$");
//		t1=System.nanoTime();
//		testMEKA();
//		System.out.println("tempo1="+(System.nanoTime()-t1)/1e9);
//	}

	public static void superPowerLucasExperiment() throws Exception { // VAMOS
																		// LA!!!
		String dataDir = "/home/lucasmello/mulan-1.4.0/data/";
		String arffFilename = dataDir + "emotions-P.arff";
		String xmlFilename = dataDir + "emotions.xml";

		MultiLabelInstances dataset = new MultiLabelInstances(arffFilename,
				xmlFilename);

		Instances data = dataset.getDataSet();

		IBk knn = new IBk(11);
		// ClassifierChain cc = new ClassifierChain(knn);
		// BinaryRelevance br = new BinaryRelevance(knn);

		List<MultiLabelLearner> mmm = new ArrayList<MultiLabelLearner>();
		PCC pcc = new PCC();
		pcc.setClassifier(knn);
		pcc.setSeed(123);
		MekaWrapperClassifier mwc = new MekaWrapperClassifier(pcc);
		mmm.add(mwc);
		// mmm.add(cc);
		// mmm.add(br);

		mwc.build(dataset);
		data.setClassIndex(6);
		for (int j = 0; j < data.numInstances(); j++) {
			MultiLabelOutput out = mwc.makePrediction(data.get(j));
			double[] pred0 = out.getConfidences();
			System.out.print(j + ": ");
			for (int i = 0; i < pred0.length; i++) {
				System.out.print(pred0[i] + " ");
			}
			System.out.println("");
		}

		// ExperimentLM exp1 = new ExperimentLM(mmm, dataset);
		// exp1.execute();
		// System.out.println(exp1.toString());
	}

	public static void testMEKA() throws Exception {
		IBk knn = new IBk(11);

		FileReader fr = new FileReader(new File(
				"/home/lucasmello/mulan-1.4.0/data/emotions-P.arff"));
		Instances insts = new Instances(fr);
		fr.close();
		insts.setClassIndex(6);

		PCC pcc = new PCC();
		pcc.setClassifier(knn);
		pcc.setSeed(123);

		pcc.buildClassifier(insts);
		for (int j = 0; j < insts.numInstances(); j++) {
			double[] pred0 = pcc.distributionForInstance(insts.get(j));
			System.out.print(j + ": ");
			for (int i = 0; i < pred0.length; i++) {
				System.out.print(pred0[i] + " ");
			}
			System.out.println("");
		}
	}
	
}
