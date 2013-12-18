package lmello;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * 
 <!-- globalinfo-start --> Class implementing the Classifier Chain (CC)
 * algorithm.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Read, Jesse, Pfahringer, Bernhard, Holmes, Geoff, Frank, Eibe: Classifier
 * Chains for Multi-label Classification. In: , 335--359, 2011.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;inproceedings{Read2011,
 *    author = {Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe},
 *    journal = {Machine Learning},
 *    number = {3},
 *    pages = {335--359},
 *    title = {Classifier Chains for Multi-label Classification},
 *    volume = {85},
 *    year = {2011}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Konstantinos Sechidis (sechidis@csd.auth.gr)
 * @author Grigorios Tsoumakas (greg@csd.auth.gr)
 * @version 2012.02.27
 */
public class CC2 extends TransformationBasedMultiLabelLearner {

	/**
	 * The new chain ordering of the label indices
	 */
	private int[] chain;

	/**
	 * Returns a string describing the classifier.
	 * 
	 * @return a string description of the classifier
	 */
	@Override
	public String globalInfo() {
		return "Class implementing the Classifier Chain (CC) algorithm."
				+ "\n\n" + "For more information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR,
				"Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe");
		result.setValue(Field.TITLE,
				"Classifier Chains for Multi-label Classification");
		result.setValue(Field.VOLUME, "85");
		result.setValue(Field.NUMBER, "3");
		result.setValue(Field.YEAR, "2011");
		result.setValue(Field.PAGES, "335--359");
		result.setValue(Field.JOURNAL, "Machine Learning");
		return result;
	}

	/**
	 * The ensemble of binary relevance models. These are Weka
	 * FilteredClassifier objects, where the filter corresponds to removing all
	 * label apart from the one that serves as a target for the corresponding
	 * model.
	 */
	protected FilteredClassifier[] ensemble;
	private Add[] addsattr;

	/**
	 * Creates a new instance using J48 as the underlying classifier
	 */
	public CC2() {
		super(new J48());
	}

	/**
	 * Creates a new instance
	 * 
	 * @param classifier
	 *            the base-level classification algorithm that will be used for
	 *            training each of the binary models
	 * @param aChain
	 */
	public CC2(Classifier classifier, int[] aChain) {
		super(classifier);
		chain = aChain;
	}

	/**
	 * Creates a new instance
	 * 
	 * @param classifier
	 *            the base-level classification algorithm that will be used for
	 *            training each of the binary models
	 */
	public CC2(Classifier classifier) {
		super(classifier);
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

	protected void buildInternal(MultiLabelInstances train) throws Exception {
		if (chain == null) {
			chain = new int[numLabels];
			for (int i = 0; i < numLabels; i++) {
				chain[i] = i;
			}
		}

		Instances trainDataset;
		numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
		trainDataset = generateData(train.getDataSet());

		for (int i = 0; i < numLabels; i++) {
			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier
					.makeCopy(baseClassifier));

			// Indices of attributes to remove first removes numLabels
			// attributes
			// the numLabels - 1 attributes and so on.
			// The loop starts from the last attribute.
			int[] indicesToRemove = new int[2 * numLabels - 1 - i];
			for (int j = 0; j < numLabels - i; j++) {
				indicesToRemove[j] = trainDataset.numAttributes() - numLabels
						+ chain[numLabels - 1 - j];
			}
			for (int j = numLabels - i; j < numLabels; j++) {
				indicesToRemove[j] = labelIndices[chain[j - numLabels + i]];
			}
			for (int j = numLabels; j < indicesToRemove.length; j++) {
				indicesToRemove[j] = labelIndices[chain[j - numLabels + i + 1]];
			}

			// for(int j=0;j<indicesToRemove.length;j++){
			// System.out.print(indicesToRemove[j]+" ");
			// }
			// System.out.println("");

			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesToRemove);
			remove.setInputFormat(trainDataset);
			remove.setInvertSelection(false);
			ensemble[i].setFilter(remove);

			trainDataset.setClassIndex(labelIndices[chain[i]]);
			debug("Bulding model " + (i + 1) + "/" + numLabels);

			ensemble[i].buildClassifier(trainDataset);

//			long t1=System.nanoTime();
			try {
				for (int k = 0; k < trainDataset.numInstances(); k++) {
					Instance inst = trainDataset.instance(k);
					double distribution[];
					distribution = ensemble[i].distributionForInstance(inst);
					int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

					// Ensure correct predictions both for class values {0,1}
					// and {1,0}
					Attribute classAttribute = ensemble[i].getFilter()
							.getOutputFormat().classAttribute();

					inst.setValue(inst.numAttributes() - numLabels + chain[i],
							distribution[classAttribute.indexOfValue("1")]);
//					inst.setValue(inst.numAttributes() - numLabels + chain[i],
//							maxIndex);
				}
			} catch (Exception e) {
				throw e;
			}
//			System.out.println("t1 (ms)="+(System.nanoTime()-t1)/1e6);
		}
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception {
		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		// Instance tempInstance = DataUtils.createInstance(instance,
		// instance.weight(), instance.toDoubleArray());
		Instance tempInstance = instance;

		for (int j = 0; j < numLabels; j++) {
			addsattr[j].input(tempInstance);
			tempInstance = addsattr[j].output();
		}

		for (int counter = 0; counter < numLabels; counter++) {
			double distribution[];
			try {
				distribution = ensemble[counter]
						.distributionForInstance(tempInstance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = ensemble[counter].getFilter()
					.getOutputFormat().classAttribute();
			bipartition[chain[counter]] = (classAttribute.value(maxIndex)
					.equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[chain[counter]] = distribution[classAttribute
					.indexOfValue("1")];
			// System.out.println(confidences[chain[counter]]);

//			 tempInstance.setValue(instance.numAttributes()+chain[counter],
//			 maxIndex);
			tempInstance.setValue(instance.numAttributes() + chain[counter],
					confidences[chain[counter]]);

		}

		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
	}
}
