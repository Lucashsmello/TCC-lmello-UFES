package lmello;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;

public class ECC2 extends EnsembleOfClassifierChains {

	/**
	 * Default constructor
	 */
	public ECC2() {
		super(new J48(), 10, true, true);
	}

	public ECC2(Classifier c, int i, boolean b, boolean c2) {
		super(c, i, b, c2);
	}

	public MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {

		MultiLabelOutput[] mloensemble = new MultiLabelOutput[numOfModels];

		for (int i = 0; i < numOfModels; i++) {
			mloensemble[i] = ensemble[i].makePrediction(instance);
		}

		return combineMLO2(mloensemble);
	}

	private MultiLabelOutput combineMLO2(MultiLabelOutput[] mloensemble) {
		double[] c = mloensemble[0].getConfidences();
		boolean[] Fbipart = new boolean[mloensemble[1].getBipartition().length];
		double[] Fconf = new double[mloensemble[1].getConfidences().length];
		double maxprod = c[0];
		Fconf[0] = c[0];
		for (int i = 1; i < Fconf.length; i++) {
			Fconf[i] = c[i];
			maxprod *= c[i] >= 0.5 ? c[i] : 1 - c[i];
		}

		for (int i = 1; i < mloensemble.length; i++) {
			double[] conf = mloensemble[i].getConfidences();
			double prod = conf[0];

			for (int j = 1; j < conf.length; j++) {
				prod *= c[j] >= 0.5 ? c[j] : 1 - c[j];
			}
			if (prod > maxprod) {
				maxprod = prod;
				for (int j = 0; j < conf.length; j++) {
					Fconf[j] = c[j];
				}
			}
		}

		for (int i = 0; i < Fconf.length; i++) {
			Fbipart[i] = Fconf[i] >= 0.5;
		}
		return new MultiLabelOutput(Fbipart, Fconf);
	}
}
