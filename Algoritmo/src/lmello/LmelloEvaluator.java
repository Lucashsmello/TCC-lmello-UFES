package lmello;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class LmelloEvaluator extends Evaluator {

	private int seed2 = 1;
	protected List<Classifier> baseclassifers = new ArrayList<>();

	public LmelloEvaluator(List<Classifier> baseclassifers) {
		if (baseclassifers == null) {
			this.baseclassifers = null;
		} else {
			this.baseclassifers.addAll(baseclassifers);
		}
	}

	/**
	 * Sets the seed for reproduction of cross-validation results
	 * 
	 * @param aSeed
	 *            seed for reproduction of cross-validation results
	 */
	public void setSeed(int aSeed) {
		seed2 = aSeed;
	}

	private void checkLearner(MultiLabelLearner learner) {
		if (learner == null) {
			throw new IllegalArgumentException(
					"Learner to be evaluated is null.");
		}
	}

	private void checkData(MultiLabelInstances data) {
		if (data == null) {
			throw new IllegalArgumentException(
					"Evaluation data object is null.");
		}
	}

	private void checkMeasures(List<Measure> measures) {
		if (measures == null) {
			throw new IllegalArgumentException(
					"List of evaluation measures to compute is null.");
		}
	}

	public MultipleEvaluation crossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, int someFolds) {
		if (baseclassifers == null) {
			return super.crossValidate(learner, data, someFolds);
		}
		checkLearner(learner);
		checkData(data);

		return innerCrossValidate(learner, data, false, null, someFolds);
	}

	public MultipleEvaluation crossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, List<Measure> measures, int someFolds) {
		if (baseclassifers == null) {
			return super.crossValidate(learner, data, measures, someFolds);
		}

		checkLearner(learner);
		checkData(data);
		checkMeasures(measures);

		return innerCrossValidate(learner, data, true, measures, someFolds);
	}

	// private Classifier getBestParameter(Instances data) throws Exception {
	// Classifier bestc = null;
	// double bestm = 0; // subset acc
	// for (Classifier c : baseclassifers) {
	// weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(
	// data);
	// eval.crossValidateModel(c, data, 10, new Random(seed2));
	// double m = eval.correct();
	// if (m > bestm) {
	// bestc = c;
	// }
	// }
	//
	// return bestc;
	// }

	private Classifier getBestParameter(MultiLabelLearner learner1,
			MultiLabelInstances data) throws Exception {
		Classifier bestc = null;
		List<Measure> meas = new ArrayList<Measure>();

		meas.add(new SubsetAccuracy());

		double bestm = 0; // subset acc
		for (Classifier c : baseclassifers) {
			Evaluator ev = new Evaluator();
			MultiLabelLearner clone = learner1.makeCopy();

			TransformationBasedMultiLabelLearner tfbmclone = (TransformationBasedMultiLabelLearner) clone;
			LmelloClassifier lmbc = (LmelloClassifier) tfbmclone
					.getBaseClassifier();
			lmbc.setClassifier(c);

			MultipleEvaluation result = ev.crossValidate(clone, data, meas, 10);
			double mean = result.getMean(new SubsetAccuracy().getName());
			if (mean > bestm) {
				bestc = c;
				bestm = mean;
			}
		}

		return bestc;
	}

	private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, boolean hasMeasures,
			List<Measure> measures, int someFolds) {
		Evaluation[] evaluation = new Evaluation[someFolds];

		Instances workingSet = new Instances(data.getDataSet());
		workingSet.randomize(new Random(seed2));
		Classifier bestc = null;
		for (int i = 0; i < someFolds; i++) {
			// System.out.println("Fold " + (i + 1) + "/" + someFolds);
			try {
				Instances train = workingSet.trainCV(someFolds, i);
				Instances test = workingSet.testCV(someFolds, i);
				MultiLabelInstances mlTrain = new MultiLabelInstances(train,
						data.getLabelsMetaData());
				MultiLabelInstances mlTest = new MultiLabelInstances(test,
						data.getLabelsMetaData());
				MultiLabelLearner clone = learner.makeCopy();
				if (clone instanceof TransformationBasedMultiLabelLearner) {
					TransformationBasedMultiLabelLearner tfbmclone = (TransformationBasedMultiLabelLearner) clone;
					Classifier bc = tfbmclone.getBaseClassifier();
					if (bc instanceof LmelloClassifier) {
						LmelloClassifier lmbc = (LmelloClassifier) bc;
						if (bestc == null) {
							bestc = getBestParameter(clone, mlTrain);
						}
						lmbc.setClassifier(bestc);
					}
				}
				clone.build(mlTrain);
				if (hasMeasures)
					evaluation[i] = evaluate(clone, mlTest, measures);
				else
					evaluation[i] = evaluate(clone, mlTest);
			} catch (Exception ex) {
				Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE,
						null, ex);
			}
		}
		MultipleEvaluation me = new MultipleEvaluation(evaluation, data);
		me.calculateStatistics();
		return me;
	}

	// public static void main(String[] args) throws Exception {
	// String dataDir = "/home/lmello/WEKA/weka-3-7-10/data/";
	// Classifier c = new IBk(3);
	// FileReader fileReader = new FileReader(new File(dataDir + "ionosphere"
	// + ".arff"));
	// Instances data = new Instances(fileReader);
	// data.setClassIndex(34);
	// weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(data);
	// eval.crossValidateModel(c, data, 10, new Random(123));
	// System.out.println(eval.correct());
	// }
	public static void main(String[] args) throws Exception {
		String dataDir = "/home/lmello/WEKA/mulan-1.4.0/data/";
		List<Classifier> ccs = new ArrayList<>();
		MultiLabelLearner mll = new BinaryRelevance(new LmelloClassifier(null));
		MultiLabelInstances dataset = new MultiLabelInstances(dataDir
				+ "emotions" + ".arff", dataDir + "emotions" + ".xml");
		ccs.add(new IBk(3));
		ccs.add(new J48());
		LmelloEvaluator ev = new LmelloEvaluator(ccs);

		MultipleEvaluation mev = ev.crossValidate(mll, dataset, 10);
		System.out.println(mev.toString());
	}
}
