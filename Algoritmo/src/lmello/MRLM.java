/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    ClassifierChain.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package lmello;

import java.util.ArrayList;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.DataUtils;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Add;

public class MRLM extends TransformationBasedMultiLabelLearner {

	/**
	 * The ensemble of binary relevance models. These are Weka
	 * FilteredClassifier objects, where the filter corresponds to removing all
	 * label apart from the one that serves as a target for the corresponding
	 * model.
	 */
	protected FilteredClassifier[][] ensemble;
	int chainSize = 2;
	BinaryRelevance br;
	Add[] addsattr;

	/**
	 * Creates a new instance
	 * 
	 * @param classifier
	 *            the base-level classification algorithm that will be used for
	 *            training each of the binary models
	 * @param aChain
	 */
	public MRLM(Classifier classifier, int chainSize) {
		super(classifier);
		this.chainSize = chainSize;
		br = new BinaryRelevance(classifier);
	}

	/**
	 * Creates a new instance
	 * 
	 * @param classifier
	 *            the base-level classification algorithm that will be used for
	 *            training each of the binary models
	 */
	public MRLM(Classifier classifier) {
		this(classifier, 2);
	}

	private void TransformInstance(Instance inst, MultiLabelOutput mlo) {
		if (mlo != null) {
			double[] confs = mlo.getConfidences();
			boolean[] bipart= mlo.getBipartition();
			for (int j = 0; j < numLabels; j++) {
//				inst.setValue(inst.numAttributes() - numLabels + j, confs[j]);
				inst.setValue(inst.numAttributes() - numLabels + j, bipart[j]?1:0);
			}
		}
	}

	private MultiLabelOutput ChainMakePrediction(int c, Instance inst,
			MultiLabelOutput prevOut) throws Exception {

		// boolean[] bipart = prevOut.getBipartition();

		// double[] bv = new double[numLabels];
		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		TransformInstance(inst, prevOut);

		for (int j = 0; j < numLabels; j++) {
			// double v = inst.value(labelIndices[j]);
			// inst.setValue(labelIndices[j], bv[j]);
			double[] distribution = ensemble[c][j]
					.distributionForInstance(inst);
			// inst.setValue(labelIndices[j], v);

			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			bipartition[j] = (maxIndex == 1) ? true : false;

			// The confidence of the label being equal to 1
			confidences[j] = distribution[1];
		}
		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
	}

	private MultiLabelOutput ChainMakePrediction(int c, Instance inst)
			throws Exception {
		return ChainMakePrediction(c, inst, null);
	}

	private FilteredClassifier constructClassifier(int labeli,
			Instances trainDataset) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(AbstractClassifier.makeCopy(baseClassifier));
		int[] indicesToRemove = new int[numLabels];
		// ensemble[c][j].setClassifier(AbstractClassifier
		// .makeCopy(baseClassifier));

		int k;
		for (k = 0; k < labeli; k++) {
			indicesToRemove[k] = labelIndices[k];
		}
		indicesToRemove[k] = trainDataset.numAttributes() - numLabels + labeli;
		for (k = labeli + 1; k < numLabels; k++) {
			indicesToRemove[k] = labelIndices[k];
		}

		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indicesToRemove);
		remove.setInputFormat(trainDataset);
		remove.setInvertSelection(false);

		fc.setFilter(remove);
		return fc;
	}

	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception {

		numLabels = train.getNumLabels();
		Instances trainData = train.getDataSet();
		ensemble = new FilteredClassifier[chainSize][numLabels];

		addsattr = new Add[numLabels];
		Instances newtrainData = trainData;

		for (int j = 0; j < numLabels; j++) {
			addsattr[j] = new Add();
			addsattr[j].setOptions(new String[] { "-T", "NUM" });

			addsattr[j].setAttributeIndex("last");
			addsattr[j].setAttributeName("labelAttr" + j);
			addsattr[j].setInputFormat(newtrainData);

			newtrainData = Filter.useFilter(newtrainData, addsattr[j]);
		}

		br.build(train);

		for (int i = 0; i < newtrainData.numInstances(); i++) {
			Instance inst = newtrainData.instance(i);
			MultiLabelOutput mlo = br.makePrediction(inst);
			TransformInstance(inst, mlo);
		}

		if (chainSize == 0) {
			return;
		}

		for (int c = 0; c < chainSize - 1; c++) {
			for (int i = 0; i < newtrainData.numInstances(); i++) {

				for (int j = 0; j < numLabels; j++) {

					ensemble[c][j] = constructClassifier(j, newtrainData);

					newtrainData.setClassIndex(labelIndices[j]);
					ensemble[c][j].buildClassifier(newtrainData);
				}
				Instance inst = newtrainData.instance(i);
				MultiLabelOutput mlo;

				mlo = ChainMakePrediction(c, inst);

				TransformInstance(inst, mlo);
			}
		}
		for (int j = 0; j < numLabels; j++) {
			ensemble[chainSize - 1][j] = constructClassifier(j, newtrainData);
			newtrainData.setClassIndex(labelIndices[j]);
			ensemble[chainSize - 1][j].buildClassifier(newtrainData);
		}

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception {
		Instance tempInstance = instance;

		for (int j = 0; j < numLabels; j++) {
			addsattr[j].input(tempInstance);
			tempInstance = addsattr[j].output();
		}

		MultiLabelOutput mlo = br.makePrediction(instance);

		for (int c = 0; c < chainSize; c++) {
			mlo = ChainMakePrediction(c, tempInstance, mlo);
		}

		return mlo;
	}
}