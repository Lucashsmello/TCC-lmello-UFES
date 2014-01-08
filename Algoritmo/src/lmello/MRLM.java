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
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

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
	private int chainSize = 2;
	private int realchainSize;
	MultiLabelLearner baseml;
	Add[] addsattr;
	double Climit = 0.1;

	List<Integer> indexs = new ArrayList<Integer>();
	private boolean instanceSelection = true;
	private boolean useTrainPropag;
	private boolean useOnlyLabels;

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
		realchainSize = chainSize;
		baseml = new BinaryRelevance(classifier);
		useTrainPropag = false;
		useOnlyLabels = false;
	}

	/**
	 * Creates a new instance
	 * 
	 * @param classifier
	 *            the base-level classification algorithm that will be used for
	 *            training each of the binary models
	 * @param aChain
	 */
	public MRLM(MultiLabelLearner baseml, Classifier classifier, int chainSize) {
		super(classifier);
		this.chainSize = chainSize;
		realchainSize = chainSize;
		this.baseml = baseml;
		useTrainPropag = false;
		useOnlyLabels = false;
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

	private void TransformInstance(Instance inst, MultiLabelOutput mlo,
			boolean useConf) {
		if (mlo != null) {
			double[] confs = mlo.getConfidences();
			boolean[] bipart = mlo.getBipartition();
			for (int j = 0; j < numLabels; j++) {
				if (useConf) {
					inst.setValue(inst.numAttributes() - numLabels + j,
							confs[j]);
				} else {
					inst.setValue(inst.numAttributes() - numLabels + j,
							bipart[j] ? 1 : 0);
				}
			}
		}
	}

	private void TransformInstance(Instance inst, MultiLabelOutput mlo) {
		TransformInstance(inst, mlo, true);
	}

	private MultiLabelOutput ChainMakePrediction(int c, Instance inst,
			MultiLabelOutput prevOut) throws Exception {

		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];
		if (c < realchainSize - 1) {
			TransformInstance(inst, prevOut);
		} else {
			TransformInstance(inst, prevOut, false);
		}
		for (int j = 0; j < numLabels; j++) {
			// double v = inst.value(labelIndices[j]);
			// inst.setValue(labelIndices[j], bv[j]);

			double[] distribution = ensemble[c][j]
					.distributionForInstance(inst);

			// if ((distribution[0] > 0.1) && (distribution[0] < 0.9)) {
			// for (int x = 0; x < distribution.length; x++) {
			// System.out.print(distribution[x] + " ");
			// }
			// System.out.println();
			// }

			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			bipartition[j] = (maxIndex == 1) ? true : false;

			// The confidence of the label being equal to 1
			confidences[j] = distribution[1];

			if (c < realchainSize - 1) {
				inst.setValue(inst.numAttributes() - numLabels + j,
						confidences[j]);
			} else {
				inst.setValue(inst.numAttributes() - numLabels + j, maxIndex);
			}
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
		int[] indicesToRemove;
		int nattrs = trainDataset.numAttributes();
		// int[] indicesToRemove = new int[nattrs-numLabels];
		// ensemble[c][j].setClassifier(AbstractClassifier
		// .makeCopy(baseClassifier));

		int k;

		if (useOnlyLabels) {
			int x;
			indicesToRemove = new int[nattrs - numLabels - 1];
			for (x = 0, k = 0; k < nattrs - numLabels; k++) {
				if (k == labelIndices[labeli]) {
					indicesToRemove[k] = trainDataset.numAttributes()
							- numLabels + labeli;
					continue;
				}
				indicesToRemove[x] = k;
				x++;
			}
		} else {
			indicesToRemove = new int[numLabels - 1];
			for (k = 0; k < labeli; k++) {
				indicesToRemove[k] = labelIndices[k];
			}
			// indicesToRemove[k] = trainDataset.numAttributes() - numLabels +
			// labeli;
			for (k = labeli + 1; k < numLabels; k++) {
				indicesToRemove[k - 1] = labelIndices[k];
			}
		}

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
		int[] indicesToRemove = new int[2 * numLabels - labeli];
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
		// k++;
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
			if (!useTrainPropag) {
				if (c > 0) {
					ensemble[c][j] = ensemble[0][j];
					continue;
				}
			}

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
	
	private boolean hit(MultiLabelOutput mlo, Instance inst) {
		boolean[] bip = mlo.getBipartition();
		for (int i = 0; i < numLabels; i++) {
			if (bip[i]) {
				if (inst.value(labelIndices[i]) == 0) {
					return false;
				}
			} else {
				if (inst.value(labelIndices[i]) == 1) {
					return false;
				}
			}
		}
		return true;
	}

	private void buildInternal2(MultiLabelInstances train) throws Exception {
		numLabels = train.getNumLabels();
		Instances trainData = train.getDataSet();
		int numinsts = trainData.numInstances();
		if (instanceSelection) {
			debug("Data size=" + numinsts);
		}

		ensemble = new FilteredClassifier[chainSize][numLabels];

		Instances newtrainData = generateData(trainData);

		// debug("Bulding model BR");
		baseml.build(train);

		indexs.clear();
		if (useTrainPropag || instanceSelection) {
			debug("Propagating Prediction 0");
			for (int i = numinsts - 1; i >= 0; i--) {

				Instance inst = newtrainData.instance(i);

				MultiLabelOutput mlo = baseml.makePrediction(inst);
				if (hit(mlo, inst)) {
					if (useTrainPropag) {
						TransformInstance(inst, mlo);
					}
					indexs.add(i);
				}
			}
		}

		if (chainSize == 0) {
			return;
		}

		for (int c = 0; c < chainSize - 1; c++) {
			if (instanceSelection) {
				for (int i = 0; i < indexs.size(); i++) {
					newtrainData.remove((int) indexs.get(i));
				}

				indexs.clear();
				numinsts = newtrainData.numInstances();
				debug("Data size=" + numinsts);

				if (!checkData(newtrainData)) {
					realchainSize = c;
					return;
				}
			}
			buildChainClassifier(c, newtrainData);
			if (useTrainPropag || instanceSelection) {
				debug("Propagating Prediction " + (c + 1));

				for (int i = numinsts - 1; i >= 0; i--) {
					Instance inst = newtrainData.instance(i);
					MultiLabelOutput mlo;

					mlo = ChainMakePrediction(c, inst);
					if (hit(mlo, inst)) {
						if (useTrainPropag) {
							TransformInstance(inst, mlo);
						}
						indexs.add(i);
					}
				}
			}
		}
		
		if (instanceSelection) {
			for (int i = 0; i < indexs.size(); i++) {
				newtrainData.remove((int) indexs.get(i));
			}
			indexs.clear();
			numinsts = newtrainData.numInstances();
			debug("Data size=" + numinsts);

			if (!checkData(newtrainData)) {
				realchainSize = chainSize - 1;
				return;
			}
		}
		realchainSize = chainSize;

		buildChainClassifier(chainSize - 1, newtrainData);
	}

	private boolean checkData(Instances data) {
		int[] countLabels = countLabels(data, true);
		for (int cl : countLabels) {
			if (cl < 11) {
				return false;
			}
		}
		countLabels = countLabels(data, false);
		for (int cl : countLabels) {
			if (cl < 11) {
				return false;
			}
		}
		return true;
	}

	int[] countLabels(Instances data, boolean has) {
		int[] clabels = new int[numLabels];
		for (int i = 0; i < clabels.length; i++) {
			clabels[i] = 0;
		}

		if (has) {
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				for (int j = data.numAttributes() - numLabels; j < data
						.numAttributes(); j++) {
					if (inst.value(j) > 0.5) {
						clabels[j - data.numAttributes() + numLabels]++;
					}
				}
			}
		} else {
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				for (int j = data.numAttributes() - numLabels; j < data
						.numAttributes(); j++) {
					if (inst.value(j) <= 0.5) {
						clabels[j - data.numAttributes() + numLabels]++;
					}
				}
			}
		}

		return clabels;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception {
		Instance tempInstance = instance;
		// System.out.println("makePredictionInternal IN");

		for (int j = 0; j < numLabels; j++) {
			addsattr[j].input(tempInstance);
			tempInstance = addsattr[j].output();
		}
		MultiLabelOutput[] mloensemble = new MultiLabelOutput[realchainSize + 1];
		mloensemble[0] = baseml.makePrediction(instance);


		for (int c = 0; c < realchainSize; c++) {
			mloensemble[c + 1] = ChainMakePrediction(c, tempInstance,
					mloensemble[c]);
		}

		// System.out.println("makePredictionInternal OUT");

		return mloensemble[realchainSize];
		// return mloensemble[chainSize];
		// return combineMLO2(mloensemble);
	}

	private MultiLabelOutput combineMLO(MultiLabelOutput[] mloensemble) {
		double[] c = mloensemble[0].getConfidences();
		boolean[] Fbipart = new boolean[mloensemble[0].getBipartition().length];
		double[] Fconf = new double[mloensemble[0].getConfidences().length];
		for (int i = 0; i < Fconf.length; i++) {
			Fconf[i] = c[i];
		}
		for (int i = 1; i < mloensemble.length; i++) {

			double[] conf = mloensemble[i].getConfidences();
			for (int j = 0; j < conf.length; j++) {
				Fconf[j] += conf[j];
			}
		}
		// for (int i = 0; i < mloensemble.length; i++) {
		//
		// double[] conf = mloensemble[i].getConfidences();
		// System.out.print(conf[i] + " ");
		// }
		// System.out.println("");
		// for (int i = 1; i < mloensemble.length; i++) {
		// double[] conf = mloensemble[i].getConfidences();
		// for (int j = 0; j < conf.length; j++) {
		// if (conf[j] > Fconf[j]) {
		// Fconf[j] = conf[j];
		// }
		// }
		// }
		for (int i = 0; i < Fconf.length; i++) {
			Fconf[i] /= mloensemble.length;
			Fbipart[i] = Fconf[i] > 0.5;
		}
		return new MultiLabelOutput(Fbipart, Fconf);
	}

	private MultiLabelOutput combineMLO2(MultiLabelOutput[] mloensemble) {

		if (realchainSize == 0) {
			return mloensemble[0];
		}
		
		double[] c = mloensemble[1].getConfidences();
		boolean[] Fbipart = new boolean[mloensemble[0].getBipartition().length];
		double[] Fconf = new double[mloensemble[0].getConfidences().length];
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

	public void setInstanceSelection(boolean t) {
		instanceSelection = t;
	}

	public void setTrainPropagation(boolean tp) {
		useTrainPropag = tp;
	}

	public void setUseOnlyLabels(boolean l) {
		useOnlyLabels = l;
	}
}