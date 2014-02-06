package lmello;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import meka.classifiers.multilabel.MCC;
import meka.classifiers.multilabel.PCC;
import weka.classifiers.trees.J48;

public class ExperimentLMTCC {
	public static void superPowerLucasExperiment() throws Exception { // VAMOS
		// LA!!!

		String dataDir = "/home/lucasmello/mulan-1.4.0/data/";
		String expDir = "/home/lucasmello/ufes/10periodo/POC2hg/Algoritmo/exps/expV3-1/";
		// String[] datasnames = new String[] { "emotions-P", "birds-P",
		// "CAL500-P", "Corel5k-P", "scene-P", "yeast-P", "medical-P",
		// "enron-P" };
		String[] datasnames = new String[] { "emotions-P","birds-P" };
		// String[] datasnames = new String[] { "enron-P", "genbase-P",
		// "rcv1subset1-P" };
		SimpleDateFormat sdffile = new SimpleDateFormat("yy-MM-dd");
		FileWriter logfile = new FileWriter(new File(expDir + "expLog2-"
				+ sdffile.format(new Date())));

		List<Classifier> baseclassifs = createBaseClassifiers();

		int i = 1;
		for (String dataname : datasnames) {
			try {
				MultiLabelInstances dataset = new MultiLabelInstances(dataDir
						+ dataname + ".arff", dataDir + dataname + ".xml");
				List<MultiLabelLearner> mmm = createMLLs2(baseclassifs, dataset);

				// configureClassifiers(baseclassifs, dataset);

				ExperimentLM exp1 = new ExperimentLM(mmm, dataset, dataname);
				exp1.setDymParameters(baseclassifs);
				exp1.setAllMeasures();
				exp1.execute();

				try {
					exp1.WriteTo(expDir + "exp" + sdffile.format(new Date())
							+ "_" + dataname);
				} catch (IOException ex) {
					System.out.println(exp1.toString());
				}
				mmm = null;
			} catch (Exception ex) {
				System.out.println("ERROR, skipping data " + dataname);
				ExperimentLMTCC.logError(logfile, ex);
			}

			String msg = "Experiment " + i + "/" + datasnames.length
					+ " Finished";
			System.out.println(msg);
			ExperimentLMTCC.log(logfile, msg);

			i++;
			System.gc();
		}

		logfile.close();
	}

	private static void logError(Writer logfile, Exception ex) {
		ex.printStackTrace();
		log(logfile, ex.toString());
	}

	private static void log(Writer logfile, String msg) {
		try {
			logfile.write(ExperimentLM.sdf.format(new Date()) + " -> " + msg
					+ "\n");
			logfile.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("LOG ERROR!!!");
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws Exception {
		long t1 = System.nanoTime();
		superPowerLucasExperiment();
		long tt = (System.nanoTime() - t1);
		if (tt < 1e11) {
			System.out.println("Tempo Total=" + tt / 1e9 + " segundos");
		} else {
			if (tt < 60e11) {
				System.out.println("Tempo Total=" + tt / 60e9 + " minutos");
			} else {
				System.out.println("Tempo Total=" + tt / 36e11 + " horas");
			}
		}
	}

	private static void configureClassifiers(List<Classifier> baseclassifs,
			MultiLabelInstances mldata) {
		for (Classifier c : baseclassifs) {
			if (c instanceof MultilayerPerceptron) {
				MultilayerPerceptron mlp = (MultilayerPerceptron) c;
				mlp.setHiddenLayers(Integer.toString(2 * mldata.getDataSet()
						.numAttributes()));
			}
		}
	}

	static List<Classifier> createBaseClassifiers() {
		List<Classifier> baseclassifs = new ArrayList<Classifier>();
		IBk knn = new IBk(5);
		SMO svm = new SMO();
		svm.setBuildLogisticModels(true);
		J48 j48 = new J48();
		// MultilayerPerceptron mlp = new MultilayerPerceptron();
		// mlp.setSeed(ExperimentLM.globalseed);
		// mlp.setTrainingTime(10);
		NaiveBayes nb = new NaiveBayes();
		weka.classifiers.functions.Logistic logi = new Logistic();
		try {
			logi.setOptions(new String[] { "-M", "10" });
			baseclassifs.add(logi);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// baseclassifs.add(mlp);

		baseclassifs.add(nb);
		baseclassifs.add(j48);
		baseclassifs.add(knn);
		baseclassifs.add(svm);

		return baseclassifs;
	}

	static List<MultiLabelLearner> createMLLs(List<Classifier> baseclassifs,
			MultiLabelInstances mldata) throws Exception {
		List<MultiLabelLearner> mmm = new ArrayList<MultiLabelLearner>();

		for (Classifier c : baseclassifs) {
			ClassifierChain cc = new ClassifierChain(c);
			// CC2 cc2 = new CC2(c);
			BinaryRelevance br = new BinaryRelevance(c);
			PCC pcc = new PCC();
			pcc.setClassifier(c);
			pcc.setSeed(ExperimentLM.globalseed);
			MRLM mrlm = new MRLM(new BinaryRelevance(c), c, 5);
			mrlm.setInstanceSelection(false);
			mrlm.setTrainPropagation(false);
			mrlm.setUseOnlyLabels(false);
			mrlm.setUseMirrorLabel(false);
			mrlm.setUseConfiability(false);
			mrlm.setChainUpdate(true);

			// MRLM mrlm2 = new MRLM(new ClassifierChain(c), c, 5);
			// mrlm2.setInstanceSelection(true);
			// mrlm2.setTrainPropagation(false);
			// mrlm2.setUseOnlyLabels(false);
			// mrlm2.setUseMirrorLabel(false);
			// mrlm2.setUseConfiability(false);
			// mrlm2.setChainUpdate(true);

			DBR dbr = new DBR(c);

			// mrlm.setDebug(true);
			MCC mcc = new MCC();
			mcc.setOptions(new String[] { "-Iy", "20" });
			mcc.setClassifier(c);
			mulan.classifier.transformation.LabelPowerset lpower = new LabelPowerset(
					c);
			lpower.setSeed(ExperimentLM.globalseed);
			lpower.setConfidenceCalculationMethod(1);
			// mulan.classifier.transformation.EnsembleOfPrunedSets eps = new
			// EnsembleOfPrunedSets(
			// 66, 10, 0.5, 2, PrunedSets.Strategy.A, 3, c);

			mmm.add(new EnsembleOfClassifierChains(c, 10, true, true));
			mmm.add(lpower);
			// mmm.add(new ECC2(c, 10, true, true));
			mmm.add(new MekaWrapperClassifier(mcc));
			if (mldata.getNumLabels() <= 10) {
				mmm.add(new MekaWrapperClassifier(pcc));
			}
			mmm.add(cc);
			// mmm.add(cc2);
			mmm.add(br);
			mmm.add(mrlm);
			// mmm.add(mrlm2);
			mmm.add(dbr);

			// mmm.add(eps);
		}

		return mmm;
	}

	static List<MultiLabelLearner> createMLLs2(List<Classifier> baseclassifs,
			MultiLabelInstances mldata) throws Exception {
		List<MultiLabelLearner> mmm = new ArrayList<MultiLabelLearner>();

		Classifier c = new LmelloClassifier(null);

		ClassifierChain cc = new ClassifierChain(c);
		BinaryRelevance br = new BinaryRelevance(c);
		PCC pcc = new PCC();
		pcc.setClassifier(c);
		pcc.setSeed(ExperimentLM.globalseed);
		MRLM mrlm = new MRLM(new BinaryRelevance(c), c, 5);
		mrlm.setInstanceSelection(false);
		mrlm.setTrainPropagation(false);
		mrlm.setUseOnlyLabels(false);
		mrlm.setUseMirrorLabel(false);
		mrlm.setUseConfiability(false);
		mrlm.setChainUpdate(true);

		DBR dbr = new DBR(c);

		MCC mcc = new MCC();
		mcc.setOptions(new String[] { "-Iy", "20" });
		mcc.setClassifier(c);
		mulan.classifier.transformation.LabelPowerset lpower = new LabelPowerset(
				c);
		lpower.setSeed(ExperimentLM.globalseed);
		lpower.setConfidenceCalculationMethod(1);

		 mmm.add(new EnsembleOfClassifierChains(c, 10, true, true));
		 mmm.add(lpower);
		 mmm.add(new MekaWrapperClassifier(mcc));
		 if (mldata.getNumLabels() <= 10) {
		 mmm.add(new MekaWrapperClassifier(pcc));
		 }
		 mmm.add(cc);
		 mmm.add(br);
		mmm.add(mrlm);
		 mmm.add(dbr);

		return mmm;
	}
}
