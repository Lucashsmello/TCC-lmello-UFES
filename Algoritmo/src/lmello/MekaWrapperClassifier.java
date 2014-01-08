package lmello;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.multilabel.MultilabelClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

public class MekaWrapperClassifier extends
		mulan.classifier.MultiLabelLearnerBase {

	private MultilabelClassifier mlclassif;
	private int numLabels;

	public MekaWrapperClassifier(
			weka.classifiers.multilabel.MultilabelClassifier mlc) {
		this.mlclassif = mlc;
	}

	/**
	 * Os primeiros atributos devem ser as classes!!!
	 */
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		numLabels = trainingSet.getNumLabels();
		Instances data = trainingSet.getDataSet();
		int tmpci = data.classIndex();
		data.setClassIndex(numLabels);
		mlclassif.buildClassifier(data);
		data.setClassIndex(tmpci);
		trainingSet = trainingSet;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {

		double[] distri = mlclassif.distributionForInstance(instance);
		boolean[] bipart = new boolean[distri.length];

		for (int i = 0; i < distri.length; i++) {
			bipart[i] = distri[i] > 0.5;
		}
		
		return new MultiLabelOutput(bipart, distri);
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public MultilabelClassifier getMultilabelClassifier(){
		return mlclassif;
	}

}
