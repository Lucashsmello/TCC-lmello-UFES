package lmello;

import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.Classifier;
import weka.core.Instances;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;

public class LmelloEvaluator extends Evaluator {

	private int seed2 = 1;
	protected List<Classifier> baseclassifers;

	public LmelloEvaluator(List<Classifier> baseclassifers) {
		this.baseclassifers = baseclassifers;
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

	/**
	 * Evaluates a {@link MultiLabelLearner} via cross-validation on given data
	 * set using given evaluation measures with defined number of folds and
	 * seed.
	 * 
	 * @param learner
	 *            the learner to be evaluated via cross-validation
	 * @param data
	 *            the multi-label data set for cross-validation
	 * @param measures
	 *            the evaluation measures to compute
	 * @param someFolds
	 * @return a {@link MultipleEvaluation} object holding the results
	 */
	public MultipleEvaluation crossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, List<Measure> measures, int someFolds) {
		checkLearner(learner);
		checkData(data);
		checkMeasures(measures);

		return innerCrossValidate(learner, data, true, measures, someFolds);
	}
	
	private Classifier getBestParameter(Instances train){
		Classifier bestc=null;
		double bestm=0; //subset acc
		weka.classifiers.evaluation.Evaluation ev=new weka.classifiers.evaluation.Evaluation(train);
		
		for(Classifier c : baseclassifers){
			ev.crossValidateModel(c, train, 10, seed2, forPredictionsPrinting)
			if()
		}
		return bestc;
	}

	private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner,
			MultiLabelInstances data, boolean hasMeasures,
			List<Measure> measures, int someFolds) {
		Evaluation[] evaluation = new Evaluation[someFolds];

		Instances workingSet = new Instances(data.getDataSet());
		workingSet.randomize(new Random(seed2));
		for (int i = 0; i < someFolds; i++) {
//			System.out.println("Fold " + (i + 1) + "/" + someFolds);
			try {
				Instances train = workingSet.trainCV(someFolds, i);
				Instances test = workingSet.testCV(someFolds, i);
				for
				
				MultiLabelInstances mlTrain = new MultiLabelInstances(train,
						data.getLabelsMetaData());
				MultiLabelInstances mlTest = new MultiLabelInstances(test,
						data.getLabelsMetaData());
				MultiLabelLearner clone = learner.makeCopy();
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

}
