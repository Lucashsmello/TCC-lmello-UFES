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
import mulan.data.MultiLabelInstances;
import weka.classifiers	.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;

public class ExperimentLMTCC_MRLM {
	public static void superPowerLucasExperiment() throws Exception { // VAMOS
		// LA!!!

		String dataDir = "/home/lucasmello/mulan-1.4.0/data/";
		String expDir = "/home/lucasmello/ufes/10periodo/POC2hg/Algoritmo/exps/expv1_MRLM/";
//		String[] datasnames = new String[] { "yeast-P" };
		 String[] datasnames = new String[] { "enron-P" };
		// String[] datasnames = new String[] { "emotions-P", "birds-P",
		// "CAL500-P", "Corel5k-P", "scene-P", "yeast-P"
		// ,"medical-P","enron-P"};
		// String[] datasnames = new String[] {
		// "emotions-P","birds-P","scene-P", "yeast-P","medical-P","Corel5k-P"};
		// String[] datasnames = new String[] {
		// "enron-P","genbase-P","rcv1subset1-P"};
		SimpleDateFormat sdffile = new SimpleDateFormat("yy-MM-dd");
		FileWriter logfile = new FileWriter(new File(expDir + "expLog"
				+ sdffile.format(new Date())));

		List<Classifier> baseclassifs = createBaseClassifiers();

		int i = 1;
		for (String dataname : datasnames) {
			try {
				MultiLabelInstances dataset = new MultiLabelInstances(dataDir
						+ dataname + ".arff", dataDir + dataname + ".xml");
				List<MultiLabelLearner> mmm = createMLLs(baseclassifs, dataset);

				configureClassifiers(baseclassifs, dataset);

				ExperimentLM exp1 = new ExperimentLM(mmm, dataset, dataname);
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
				ExperimentLMTCC_MRLM.logError(logfile, ex);
			}

			String msg = "Experiment " + i + "/" + datasnames.length
					+ " Finished";
			System.out.println(msg);
			ExperimentLMTCC_MRLM.log(logfile, msg);

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
		IBk knn = new IBk(11);
		SMO svm = new SMO();
		svm.setBuildLogisticModels(true);
		J48 j48 = new J48();
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.setSeed(ExperimentLM.globalseed);
		mlp.setTrainingTime(10);
		NaiveBayes nb = new NaiveBayes();
		weka.classifiers.functions.Logistic logi = new Logistic();
		// try {
		// logi.setOptions(new String[] { "-M", "10" });
		// baseclassifs.add(logi);
		// } catch (Exception e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }

		// baseclassifs.add(mlp);

		// baseclassifs.add(nb);
		baseclassifs.add(j48);
		baseclassifs.add(knn);
//		baseclassifs.add(svm);

		return baseclassifs;
	}

	static List<MultiLabelLearner> createMLLs(List<Classifier> baseclassifs,
			MultiLabelInstances mldata) throws Exception {
		List<MultiLabelLearner> mmm = new ArrayList<MultiLabelLearner>();

		for (Classifier c : baseclassifs) {
			for (int i = 0; i <= 5; i++) {
				mmm.add(createMRLM(c, i));
			}
		}

		return mmm;
	}

	private static MRLM createMRLM(Classifier c, int chainsize) {
		MRLM mrlm = new MRLM(new BinaryRelevance(c), c, chainsize);
		mrlm.setInstanceSelection(false);
		mrlm.setTrainPropagation(false);
		mrlm.setUseOnlyLabels(false);
		mrlm.setUseMirrorLabel(false);
		mrlm.setUseConfiability(false);
		mrlm.setChainUpdate(true);

		return mrlm;
	}
}
