package lmello;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LmelloClassifier extends AbstractClassifier{
	Classifier classif;

	public LmelloClassifier(Classifier c) {
		classif = c;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		classif.buildClassifier(data);

	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {

		return classif.classifyInstance(instance);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return classif.distributionForInstance(instance);
	}

	@Override
	public Capabilities getCapabilities() {
		return classif.getCapabilities();
	}

	public void setClassifier(Classifier c) {
		classif = c;
	}

	public Classifier getClassifier() {
		return classif;
	}
}
