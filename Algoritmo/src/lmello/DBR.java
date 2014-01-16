package lmello;

import weka.classifiers.Classifier;

public class DBR extends MRLM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DBR(Classifier classifier) {
		super(classifier, 1);
		setUseConfiability(false);
		setUseMirrorLabel(false);
		setInstanceSelection(false);
		setTrainPropagation(false);
		setUseOnlyLabels(false);
		setChainUpdate(false);
	}

}
