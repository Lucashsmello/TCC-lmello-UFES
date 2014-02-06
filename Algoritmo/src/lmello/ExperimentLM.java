package lmello;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import meka.classifiers.multilabel.MultilabelClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

public class ExperimentLM {
	public static final SimpleDateFormat sdf = new SimpleDateFormat(
			"yyyy-MM-dd HH:mm:ss");

	private List<MultiLabelLearner> mlls;
	private MultiLabelInstances mldata;
	private boolean executed = false;
	private Date begin;
	private Date end;
	private List<Long> timeExec;
	private List<Long> timeExec2;
	private long totaltimeExec;
	private List<MultipleEvaluation> results;
	private List<Measure> measures;
	final int numFolds = 10;
	private boolean useCSVMethodName = true;
	private String dataname = "???";

	private int rotationseed = 4;

	private List<Classifier> dym_parameters = null;

	private List<Integer> numattrssub;

	public static int globalseed = 1;

	public ExperimentLM(MultiLabelInstances mldata) {
		mlls = new ArrayList<MultiLabelLearner>();
		this.mldata = mldata;
		results = new ArrayList<MultipleEvaluation>();
		timeExec = new ArrayList<Long>();
		timeExec2 = new ArrayList<Long>();
		numattrssub = new ArrayList<Integer>();

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

	public void setAllMeasures() {
		// add example-based measures
		measures.clear();
		measures.add(new HammingLoss());
		measures.add(new SubsetAccuracy());
		measures.add(new ExampleBasedPrecision());
		measures.add(new ExampleBasedRecall());
		measures.add(new ExampleBasedFMeasure());
		measures.add(new ExampleBasedAccuracy());
		measures.add(new ExampleBasedSpecificity());
		// add label-based measures
		int numOfLabels = mldata.getNumLabels();
		measures.add(new MicroPrecision(numOfLabels));
		measures.add(new MicroRecall(numOfLabels));
		measures.add(new MicroFMeasure(numOfLabels));
		measures.add(new MicroSpecificity(numOfLabels));
		measures.add(new MacroPrecision(numOfLabels));
		measures.add(new MacroRecall(numOfLabels));
		measures.add(new MacroFMeasure(numOfLabels));
		measures.add(new MacroSpecificity(numOfLabels));

		// add ranking-based measures if applicable

		// add ranking based measures
		measures.add(new AveragePrecision());
		measures.add(new Coverage());
		measures.add(new OneError());
		measures.add(new IsError());
		measures.add(new ErrorSetSize());
		measures.add(new RankingLoss());

		// add confidence measures if applicable
		measures.add(new MeanAveragePrecision(numOfLabels));
		// measures.add(new GeometricMeanAveragePrecision(numOfLabels));
		// measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
		// measures.add(new
		// GeometricMeanAverageInterpolatedPrecision(numOfLabels,
		// 10));
		measures.add(new MicroAUC(numOfLabels));
		// measures.add(new MacroAUC(numOfLabels));

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

		Thread currentThread = Thread.currentThread();
		long threadId = currentThread.getId();
		ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();

		PrintStream nothingStream;
		PrintStream stdout = System.out;
		Instances data = mldata.getDataSet();
		int tmpci = data.classIndex();
		try {
			nothingStream = new PrintStream(new File("/tmp/nothinglog"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			nothingStream = System.out;
		}

		LmelloEvaluator eval = new LmelloEvaluator(dym_parameters);
		eval.setSeed(rotationseed);
		begin = new Date();

		totaltimeExec = System.nanoTime();

		int nm = 1;
		for (int i = 0; i < mlls.size(); i++) {
			// for (MultiLabelLearner mll : mlls) {
			MultiLabelLearner mll = mlls.get(i);
			long timeelapsed, timeelapsed2;
			try {

				System.setOut(nothingStream);

				if (mll instanceof MekaWrapperClassifier) {

					data.setClassIndex(mldata.getNumLabels());

					timeelapsed = System.nanoTime();
					timeelapsed2 = threadMXBean.getThreadCpuTime(threadId);
					results.add(eval.crossValidate(mll, mldata, measures,
							numFolds));
					timeelapsed2 = threadMXBean.getThreadCpuTime(threadId)
							- timeelapsed2;
					timeelapsed = System.nanoTime() - timeelapsed;

					data.setClassIndex(tmpci);
				} else {
					timeelapsed = System.nanoTime();
					timeelapsed2 = threadMXBean.getThreadCpuTime(threadId);
					results.add(eval.crossValidate(mll, mldata, measures,
							numFolds));
					timeelapsed2 = threadMXBean.getThreadCpuTime(threadId)
							- timeelapsed2;
					timeelapsed = System.nanoTime() - timeelapsed;

				}

				numattrssub.add(eval.getNumAttrsSubmitted());

				System.setOut(stdout);
				System.out.println(getMethodDescription(mll) + " FINISHED ("
						+ (100 * nm / mlls.size()) + "% completed)");
				timeExec.add(timeelapsed);
				timeExec2.add(timeelapsed2);
			} catch (OutOfMemoryError outm) {
				System.setOut(stdout);
				System.out.println(outm.toString());
				mlls.set(i, new OOMmethod(getMethodAbbrv(mll)));
				mll = null;
				System.gc();
			}
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

		s += "GLOBAL SEED: " + globalseed + "\n";
		s += "METODO DE AVALIAÇÃO: rotation, " + numFolds + " folds, seed="
				+ rotationseed + "\n";
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

		s += "Tempo(seg);";
		s += "Tempo2(seg);";
		s += "FeatsSub;";
		s += "FeatsSub(perInst);";
		s += "FeatsSub(perInstFeat);\n";
		for (int i = 0; i < results.size(); i++) {
			if (useCSVMethodName) {
				s += getMethodAbbrv(mlls.get(i)) + ";";
			}
			s += results.get(i).toCSV();
			s += ((double) timeExec.get(i)) / 1e9 + ";";
			s += ((double) timeExec2.get(i)) / 1e9 + ";";
			s += numattrssub.get(i) + ";";
			s += (float)(numattrssub.get(i)) / mldata.getNumInstances() + ";";
			s += ((float)(numattrssub.get(i)) / mldata.getNumInstances()) / (mldata.getDataSet().numAttributes()-mldata.getNumLabels()) + ";\n";
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

		if (ml instanceof OOMmethod) {
			return ml.toString();
		}

		String s = ml.getClass().getSimpleName();
		if (ml instanceof MRLM) {
			MRLM mrlm = (MRLM) ml;
			if (mrlm.isInstanceSelection()) {
				s += "-I";
			}
			s += "-" + mrlm.getChainSize();
		}

		if (ml instanceof TransformationBasedMultiLabelLearner) {
			s += " [";
			TransformationBasedMultiLabelLearner tbml = (TransformationBasedMultiLabelLearner) ml;
			Classifier c = tbml.getBaseClassifier();
			if (c instanceof LmelloClassifier) {
				c = ((LmelloClassifier) c).getClassifier();
			}

			// if (c instanceof LmelloClassifier) {
			// c = ((LmelloClassifier) c).getClassifier();
			// s += "baseClassifier=LMC -> " + c.getClass().getSimpleName();
			// } else {
			s += "baseClassifier=" + c.getClass().getSimpleName();
			// }
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

		if (ml instanceof MRLM) {
			MRLM mrlm = (MRLM) ml;
			if (mrlm.isInstanceSelection()) {
				s += "-I";
			}
			s += "-" + mrlm.getChainSize();
		}

		if (ml instanceof TransformationBasedMultiLabelLearner) {
			TransformationBasedMultiLabelLearner tbml = (TransformationBasedMultiLabelLearner) ml;
			Classifier c = tbml.getBaseClassifier();
			if (c instanceof LmelloClassifier) {
				c = ((LmelloClassifier) c).getClassifier();
			}

			if (ml instanceof ClassifierChain) {
				s = "CC";
			}

			if (ml instanceof BinaryRelevance) {
				s = "BR";
			}

			if (ml instanceof EnsembleOfClassifierChains) {
				if (ml instanceof ECC2) {
					s = "ECC2";
				} else {
					s = "ECC";
				}
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

	public void setDymParameters(List<Classifier> dp) {
		dym_parameters = dp;
	}

	private class OOMmethod extends MultiLabelLearnerBase {

		private String methodname;

		public OOMmethod(String methodname) {
			this.methodname = methodname;
		}

		@Override
		protected void buildInternal(MultiLabelInstances trainingSet)
				throws Exception {
			// TODO Auto-generated method stub

		}

		@Override
		protected MultiLabelOutput makePredictionInternal(Instance instance)
				throws Exception, InvalidDataException {
			// TODO Auto-generated method stub
			return null;
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

		public String toString() {
			return "OOM-" + methodname;
		}

	}

}
