package lmello;

import mulan.classifier.transformation.BinaryRelevance;
import weka.classifiers.Classifier;

public class DBR extends MRLM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DBR(Classifier classifier) {
		super( new BinaryRelevance(classifier),classifier, 1);
		setUseConfiability(false);
		setUseMirrorLabel(false);
		setInstanceSelection(false);
		setTrainPropagation(false);
		setUseOnlyLabels(false);
		setChainUpdate(false);
	}

}
