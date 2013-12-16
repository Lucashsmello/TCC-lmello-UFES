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
 */

/*
 *    MRLM.java
 *    Copyright (C) 2009-2012 UFES
 */
package lmello;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
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
import weka.filters.unsupervised.instance.RemoveRange;

public class MRLM extends TransformationBasedMultiLabelLearner {

	double mstime(long t) {
		return (System.nanoTime() - t) / 1e6;
	}

	/**
	 * The ensemble of binary relevance models. These are Weka
	 * FilteredClassifier objects, where the filter corresponds to removing all
	 * label apart from the one that serves as a target for the corresponding
	 * model.
	 */
	protected FilteredClassifier[][] ensemble;
	int chainSize = 2;
	MultiLabelLearner br;
	Add[] addsattr;
	double Climit = 0.1;

	List<Integer> indexs = new ArrayList<Integer>();

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
			boolean[] bipart = mlo.getBipartition();
			for (int j = 0; j < numLabels; j++) {
				// inst.setValue(inst.numAttributes() - numLabels + j,
				// confs[j]);
				inst.setValue(inst.numAttributes() - numLabels + j,
						bipart[j] ? 1 : 0);
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
		int[] indicesToRemove = new int[numLabels - 1];
		// int nattrs=trainDataset.numAttributes();
		// int[] indicesToRemove = new int[nattrs-numLabels];
		// ensemble[c][j].setClassifier(AbstractClassifier
		// .makeCopy(baseClassifier));

		int k;
		for (k = 0; k < labeli; k++) {
			indicesToRemove[k] = labelIndices[k];
		}
		// indicesToRemove[k] = trainDataset.numAttributes() - numLabels +
		// labeli;
		for (k = labeli + 1; k < numLabels; k++) {
			indicesToRemove[k - 1] = labelIndices[k];
		}
		// for (k = 0; k < nattrs-numLabels;k++) {
		// if(k==labelIndices[labeli]){
		// indicesToRemove[k] = trainDataset.numAttributes() - numLabels +
		// labeli;
		// continue;
		// }
		// indicesToRemove[k] = k;
		// }

		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indicesToRemove);
		remove.setInputFormat(trainDataset);
		remove.setInvertSelection(false);

		fc.setFilter(remove);
		return fc;
	}

	private FilteredClassifier constructClassifier2(int labeli,
			Instances trainDataset) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(AbstractClassifier.makeCopy(baseClassifier));
		int[] indicesToRemove = new int[2 * numLabels - labeli - 1];
		// int nattrs=trainDataset.numAttributes();
		// int[] indicesToRemove = new int[nattrs-numLabels];
		// ensemble[c][j].setClassifier(AbstractClassifier
		// .makeCopy(baseClassifier));

		int k;
		for (k = 0; k < labeli; k++) {
			indicesToRemove[k] = labelIndices[k];
		}
		// indicesToRemove[k] = trainDataset.numAttributes() - numLabels +
		// labeli;
		for (; k < numLabels - 1; k++) {
			indicesToRemove[k] = labelIndices[k + 1];
		}
		for (int i = trainDataset.numAttributes() - numLabels + labeli; i < trainDataset
				.numAttributes(); i++) {
			indicesToRemove[k] = i;
			k++;
		}
		// for (k = 0; k < nattrs-numLabels;k++) {
		// if(k==labelIndices[labeli]){
		// indicesToRemove[k] = trainDataset.numAttributes() - numLabels +
		// labeli;
		// continue;
		// }
		// indicesToRemove[k] = k;
		// }

		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indicesToRemove);
		remove.setInputFormat(trainDataset);
		remove.setInvertSelection(false);

		fc.setFilter(remove);
		return fc;
	}

	private void buildChainClassifier(int c, Instances newtrainData)
			throws Exception {
		for (int j = 0; j < numLabels; j++) {
			ensemble[c][j] = constructClassifier(j, newtrainData);
			newtrainData.setClassIndex(labelIndices[j]);
			// debug("Bulding model " + (c * numLabels + j + 1) + "/" +
			// numLabels
			// * chainSize);

			// RemoveRange rr=new RemoveRange();
			// rr.setInputFormat(newtrainData);
			// String irange="";
			// int i;
			// for (i = 0; i < indexs.size()-1; i++) {
			// int index = indexs.get(i);
			// irange+=index+",";
			// }
			// irange+=indexs.get(i);
			// rr.setInstancesIndices(irange);
			// rr.setInvertSelection(true);
			//
			// for(i=0;i<newtrainData.numInstances();i++){
			// rr.input(newtrainData.instance(i));
			// }
			// rr.batchFinished();

			ensemble[c][j].buildClassifier(newtrainData);
		}
	}

	private Instances generateData(Instances trainData) throws Exception {
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
		for (int i = 0; i < newtrainData.numInstances(); i++) {
			Instance inst = newtrainData.instance(i);
			for (int j = 0; j < numLabels; j++) {
				inst.setValue(j + trainData.numAttributes(),
						inst.value(labelIndices[j]));
			}
		}
		return newtrainData;
	}

	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception {
		buildInternal2(train);
	}

	protected void buildInternal1(MultiLabelInstances train) throws Exception {
		numLabels = train.getNumLabels();
		Instances trainData = train.getDataSet();
		int numinsts = trainData.numInstances();

		ensemble = new FilteredClassifier[chainSize][numLabels];

		Instances newtrainData = generateData(trainData);

		// debug("Bulding model BR");
		br.build(train);

		debug("Propagating Prediction 0");

		// for (int i = 0; i < numinsts; i++) {
		//
		// Instance inst = newtrainData.instance(i);
		//
		// MultiLabelOutput mlo = br.makePrediction(inst);
		// TransformInstance(inst, mlo);
		// }

		if (chainSize == 0) {
			return;
		}

		for (int c = 0; c < chainSize - 1; c++) {
			buildChainClassifier(c, newtrainData);

			debug("Propagating Prediction " + (c + 1));

			// int toremove = (numinsts / chainSize);
			// indexs.clear();
			// for (int i = 0; i < numinsts; i++) {
			// indexs.add(i);
			// }
			// Collections.shuffle(indexs);
			// indexs=indexs.subList(0, toremove);
			//
			// for (int i = 0; i < toremove; i++) {
			// newtrainData.remove(0);
			// }

			// numinsts = newtrainData.numInstances();
			// System.out.println(numinsts+" | "+toremove+"| "+indexs.size());
			// for (int i = 0; i < numinsts / 10; i++) {
			for (int i = 0; i < numinsts; i++) {
				Instance inst = newtrainData.instance(i);
				MultiLabelOutput mlo;

				mlo = ChainMakePrediction(c, inst);

				TransformInstance(inst, mlo);
			}
		}

		buildChainClassifier(chainSize - 1, newtrainData);
	}

	protected void buildInternal2(MultiLabelInstances train) throws Exception {
		numLabels = train.getNumLabels();
		Instances trainData = train.getDataSet();
		int numinsts = trainData.numInstances();

		ensemble = new FilteredClassifier[chainSize][numLabels];

		Instances newtrainData = generateData(trainData);

		// debug("Bulding model BR");
		br.build(train);

		debug("Propagating Prediction 0");

		// // indexs.clear();
		for (int i = 0; i < numinsts; i++) {

			Instance inst = newtrainData.instance(i);

			MultiLabelOutput mlo = br.makePrediction(inst);
			// double[] confs = mlo.getConfidences();
			// int j;
			// for (j = 0; j < confs.length; j++) {
			// if (Math.abs(0.5 - (confs[j])) < Climit) {
			// // indexs.add(i);
			// break;
			// }
			// }
			// if (j == confs.length) {
			TransformInstance(inst, mlo);
			// }

		}

		if (chainSize == 0) {
			return;
		}

		for (int c = 0; c < chainSize - 1; c++) {
			buildChainClassifier(c, newtrainData);

			debug("Propagating Prediction " + (c + 1));

			// int toremove = (numinsts / chainSize);
			// indexs.clear();
			// for (int i = 0; i < numinsts; i++) {
			// indexs.add(i);
			// }
			// Collections.shuffle(indexs);
			// indexs=indexs.subList(0, toremove);
			//
			// for (int i = 0; i < toremove; i++) {
			// newtrainData.remove(0);
			// }

			// numinsts = newtrainData.numInstances();
			// System.out.println(numinsts+" | "+toremove+"| "+indexs.size());
			// for (int i = 0; i < numinsts / 10; i++) {
			for (int i = 0; i < numinsts; i++) {
				Instance inst = newtrainData.instance(i);
				MultiLabelOutput mlo;

				mlo = ChainMakePrediction(c, inst);

				double[] confs = mlo.getConfidences();

				// int j;
				// for (j = 0; j < confs.length; j++) {
				// if (Math.abs(0.5 - confs[j]) < Climit) {
				// // indexs.add(i);
				// // System.out.println("N Transforma " + i);
				// break;
				// }
				// }
				// if (j == confs.length) {
				//
				TransformInstance(inst, mlo);
				// }
			}
		}

		buildChainClassifier(chainSize - 1, newtrainData);
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception {
		Instance tempInstance = instance;

		for (int j = 0; j < numLabels; j++) {
			addsattr[j].input(tempInstance);
			tempInstance = addsattr[j].output();
		}
		MultiLabelOutput[] mloensemble = new MultiLabelOutput[chainSize + 1];
		mloensemble[0] = br.makePrediction(instance);

		for (int c = 0; c < chainSize; c++) {

			// mlo = ChainMakePrediction(c, tempInstance, mlo);
			mloensemble[c + 1] = ChainMakePrediction(c, tempInstance,
					mloensemble[c]);
		}

//		return mloensemble[chainSize];
		 return combineMLO2(mloensemble);
	}

	private MultiLabelOutput combineMLO(MultiLabelOutput[] mloensemble) {
		double[] c = mloensemble[0].getConfidences();
		boolean[] Fbipart = new boolean[mloensemble[0].getBipartition().length];
		double[] Fconf = new double[mloensemble[0].getConfidences().length];
		for (int i = 0; i < Fconf.length; i++) {
			Fconf[i] = c[i];
		}
		// for (int i = 1; i < mloensemble.length; i++) {
		// double[] conf = mloensemble[i].getConfidences();
		// for (int j = 0; j < conf.length; j++) {
		// Fconf[j] += conf[j];
		// }
		// }
		for (int i = 1; i < mloensemble.length; i++) {
			double[] conf = mloensemble[i].getConfidences();
			for (int j = 0; j < conf.length; j++) {
				if (conf[j] > Fconf[j]) {
					Fconf[j] = conf[j];
				}
			}
		}
		for (int i = 0; i < Fconf.length; i++) {
			// Fconf[i] /= mloensemble.length;
			Fbipart[i] = Fconf[i] > 0.5;
		}
		return new MultiLabelOutput(Fbipart, Fconf);
	}

	private MultiLabelOutput combineMLO2(MultiLabelOutput[] mloensemble) {
		double[] c = mloensemble[1].getConfidences();
		boolean[] Fbipart = new boolean[mloensemble[1].getBipartition().length];
		double[] Fconf = new double[mloensemble[1].getConfidences().length];
		double maxprod = c[0];
		Fconf[0] = c[0];
		for (int i = 1; i < Fconf.length; i++) {
			Fconf[i] = c[i];
			maxprod *= c[i] >= 0.5 ? c[i] : 1 - c[i];
		}

		for (int i = 2; i < mloensemble.length; i++) {
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
			Fbipart[i] = Fconf[i] > 0.5;
		}
		return new MultiLabelOutput(Fbipart, Fconf);
	}
}