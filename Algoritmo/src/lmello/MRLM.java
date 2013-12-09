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

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Add;

public class MRLM extends TransformationBasedMultiLabelLearner {

    /**
     * The new chain ordering of the label indices
     */
    private int[] chain;
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[][] ensemble;
    int chainSize = 2;

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain
     */
    public MRLM(Classifier classifier, int chainSize) {
        super(classifier);
        this.chainSize = chainSize;
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public MRLM(Classifier classifier) {
        super(classifier);
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        if (chain == null) {
            chain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                chain[i] = i;
            }
        }


        Instances trainDataset;
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[chainSize][numLabels];
        trainDataset = train.getDataSet();

        for (int i = 0; i < chainSize; i++) {
            for (int j = 0; j < numLabels; j++) {
                ensemble[i][j] = new FilteredClassifier();
                ensemble[i][j].setClassifier(AbstractClassifier.makeCopy(baseClassifier));


//            Add addattr=new Add();
//            addattr

//            Remove remove = new Remove();
//            remove.setAttributeIndicesArray(indicesToRemove);
//            remove.setInputFormat(trainDataset);
//            remove.setInvertSelection(false);
//            ensemble[i].setFilter(remove);

                trainDataset.setClassIndex(labelIndices[j]);
                debug("Bulding model " + (j + 1) + "/" + numLabels);
                ensemble[i][j].buildClassifier(trainDataset);
            }
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        for (int i = 0; i < chainSize; i++) {
            for (int j = 0; j < numLabels; j++) {
                double distribution[];
                try {
                    distribution = ensemble[i][j].distributionForInstance(tempInstance);
                } catch (Exception e) {
                    System.out.println(e);
                    return null;
                }
                int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

                // Ensure correct predictions both for class values {0,1} and {1,0}
                Attribute classAttribute = ensemble[i][j].getFilter().getOutputFormat().classAttribute();
                bipartition[j] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

                // The confidence of the label being equal to 1
                confidences[j] = distribution[classAttribute.indexOfValue("1")];

//                tempInstance.setValue(labelIndices[chain[j]], maxIndex);

            }
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}