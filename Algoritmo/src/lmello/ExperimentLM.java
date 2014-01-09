package lmello;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Writer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.multilabel.MCC;
import weka.classifiers.multilabel.MultilabelClassifier;
import weka.classifiers.multilabel.PCC;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ExperimentLM {
	public static final SimpleDateFormat sdf = new SimpleDateFormat(
			"yyyy-MM-dd HH:mm:ss");

	private List<MultiLabelLearner> mlls;
	private MultiLabelInstances mldata;
	private boolean executed = false;
	private Date begin;
	private Date end;
	private List<Long> timeExec;
	private long totaltimeExec;
	private List<MultipleEvaluation> results;
	private List<Measure> measures;
	int numFolds = 10;
	private boolean useCSVMethodName = true;
	private String dataname = "???";

	public static int globalseed = 123;

	public ExperimentLM(MultiLabelInstances mldata) {
		mlls = new ArrayList<MultiLabelLearner>();
		this.mldata = mldata;
		results = new ArrayList<MultipleEvaluation>();
		timeExec = new ArrayList<Long>();

		measures = new ArrayList<Measure>();
		measures.add(new HammingLoss());
		measures.add(new SubsetAccuracy());
		measures.add(new mulan.evaluation.measure.RankingLoss());
		measures.add(new mulan.evaluation.measure.AveragePrecision());
	}

	public ExperimentLM(MultiLabelLearner ml, MultiLabelInstances mldata) {
		this(mldata);
		mlls.add(ml);
	}

	public ExperimentLM(List<MultiLabelLearner> mls, MultiLabelInstances mldata) {
		this(mldata);
		mlls.addAll(mls);
	}

	public ExperimentLM(List<MultiLabelLearner> mls,
			MultiLabelInstances mldata, String dataname) {
		this(mls, mldata);
		setDataName(dataname);
	}

	public void addMethod(MultiLabelLearner ml) {
		mlls.add(ml);
	}

	public void setDataName(String dname) {
		dataname = dname;
	}

	public void execute() {
		if (isExecuted()) {
			System.out.println("Experiment already executed!");
		}

		PrintStream nothingStream;
		try {
			nothingStream = new PrintStream(new File("/tmp/nothinglog"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			nothingStream = System.out;
		}

		Evaluator eval = new Evaluator();
		eval.setSeed(123);
		begin = new Date();

		totaltimeExec = System.nanoTime();

		int nm = 1;
		for (MultiLabelLearner mll : mlls) {
			PrintStream stdout = System.out;
			System.setOut(nothingStream);
			long timeelapsed;
			if (mll instanceof MekaWrapperClassifier) {
				Instances data = mldata.getDataSet();
				int tmpci = data.classIndex();
				data.setClassIndex(mldata.getNumLabels());

				timeelapsed = System.nanoTime();
				results.add(eval.crossValidate(mll, mldata, measures, numFolds));
				timeelapsed = System.nanoTime() - timeelapsed;

				data.setClassIndex(tmpci);
			} else {
				timeelapsed = System.nanoTime();
				results.add(eval.crossValidate(mll, mldata, measures, numFolds));
				timeelapsed = System.nanoTime() - timeelapsed;
			}
			System.setOut(stdout);
			System.out.println(getMethodDescription(mll) + " FINISHED ("
					+ (100 * nm / mlls.size()) + "% completed)");
			timeExec.add(timeelapsed);
			nm++;
		}

		totaltimeExec = System.nanoTime() - totaltimeExec;
		end = new Date();
		executed = true;
		if (nothingStream != System.out) {
			nothingStream.close();
		}
	}

	public String ExecuteTime() {
		if (totaltimeExec < 1e9) {
			return new String(((double) totaltimeExec) / 1e6 + " miliseconds");
		}
		if (totaltimeExec < 1e11) {
			return new String(secondsElapsed() + " seconds");
		}
		if (totaltimeExec < 60e11) {
			return new String(((double) totaltimeExec) / 60e9 + " minutes");
		}

		return new String(((double) totaltimeExec) / 36e11 + " hours");
	}

	public double secondsElapsed() {
		return ((double) totaltimeExec) / 1e9;
	}

	public List<MultipleEvaluation> getResults() {
		return results;
	}

	public boolean isExecuted() {
		return executed;
	}

	public Date getBegin() {
		return begin;
	}

	public Date getEnd() {
		return end;
	}

	public boolean isUseCSVMethodName() {
		return useCSVMethodName;
	}

	public void setUseCSVMethodName(boolean useCSVMethodName) {
		this.useCSVMethodName = useCSVMethodName;
	}

	public String toString() {
		String username;
		try {
			username = System.getProperty("user.name");
		} catch (Exception ex) {
			username = "???";
		}
		String machinename, machineaddress;

		try {
			java.net.InetAddress localmach = java.net.InetAddress
					.getLocalHost();
			machinename = localmach.getHostName();
			machineaddress = localmach.getHostAddress();
		} catch (Exception e) {
			machinename = "???";
			machineaddress = "???";
		}

		String s = "EXPERIMENTO INICIADO EM  " + sdf.format(begin) + "\n";
		s += "EXPERIMENTO TERMINADO EM " + sdf.format(end) + "\n";
		s += "TEMPO TOTAL DE EXECUCAO: " + ExecuteTime() + "\n";
		s += "Nome/Endereco da Maquina: " + machinename + " | "
				+ machineaddress + "\n";
		s += "Nome do Usuario: " + username + "\n\n";

		s += "BASE DE DADOS: " + dataname + ", " + mldata.getNumInstances()
				+ " instancias, " + mldata.getDataSet().numAttributes()
				+ " atributos, " + mldata.getNumLabels() + " rótulos" + "\n";
		s += "ALGORITMOS:\n";
		for (MultiLabelLearner ml : mlls) {
			s += "   " + ExperimentLM.getMethodDescription(ml) + "\n";
		}
		s += "RESULTADOS: \n";

		// if (mlls.size() > 1) {
		if (useCSVMethodName) {
			s += " ;";
		}
		for (Measure m : measures) {
			s += m.getName() + ";";
		}

		s += "Tempo(seg) \n";
		for (int i = 0; i < results.size(); i++) {
			if (useCSVMethodName) {
				s += getMethodAbbrv(mlls.get(i)) + ";";
			}
			s += results.get(i).toCSV();
			s += ((double) timeExec.get(i)) / 1e9 + ";\n";
		}
		// } else {
		// s += results.get(0).toString();
		// }

		return s;
	}

	public void WriteTo(String file) throws IOException {
		if (!isExecuted()) {
			execute();
		}
		FileWriter fw = new FileWriter(new File(file));
		fw.write(toString());
		fw.close();
	}

	static String getMethodDescription(MultiLabelLearner ml) {
		// String s = ml.getClass().getName();
		String s = ml.getClass().getSimpleName();

		if (ml instanceof TransformationBasedMultiLabelLearner) {
			s += " [";
			TransformationBasedMultiLabelLearner tbml = (TransformationBasedMultiLabelLearner) ml;
			Classifier c = tbml.getBaseClassifier();
			s += "baseClassifier=" + c.getClass().getSimpleName();
			if (c instanceof IBk) {
				s += "(" + ((IBk) c).getKNN() + ")";
			}
			s += "]";
		} else {
			if (ml instanceof MekaWrapperClassifier) {
				MekaWrapperClassifier mwc = (MekaWrapperClassifier) ml;
				MultilabelClassifier mlcmeka = mwc.getMultilabelClassifier();
				s = "meka.";
				s += mlcmeka.getClass().getSimpleName() + " [";
				Classifier c = mlcmeka.getClassifier();

				s += "baseClassifier=" + c.getClass().getSimpleName();
				if (c instanceof IBk) {
					s += "(" + ((IBk) c).getKNN() + ")";
				}
			}
			s += "]";
		}

		return s;
	}

	static String getMethodAbbrv(MultiLabelLearner ml) {
		String s = ml.getClass().getSimpleName();

		if (ml instanceof TransformationBasedMultiLabelLearner) {
			TransformationBasedMultiLabelLearner tbml = (TransformationBasedMultiLabelLearner) ml;
			Classifier c = tbml.getBaseClassifier();

			if (ml instanceof ClassifierChain) {
				s = "CC";
			}

			if (ml instanceof BinaryRelevance) {
				s = "BR";
			}
			if (ml instanceof EnsembleOfClassifierChains) {
				s = "ECC";
			}

			if (c instanceof MultilayerPerceptron) {
				s += " (ANN)";
			} else {
				s += " (" + c.getClass().getSimpleName() + ")";
			}
		} else {
			if (ml instanceof MekaWrapperClassifier) {
				MekaWrapperClassifier mwc = (MekaWrapperClassifier) ml;
				MultilabelClassifier mlcmeka = mwc.getMultilabelClassifier();
				Classifier c = mlcmeka.getClassifier();

				s = mlcmeka.getClass().getSimpleName();
				s += " (" + c.getClass().getSimpleName() + ")";
			}
		}

		return s;
	}

}
