package ESKDBTest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import org.apache.commons.math3.random.MersenneTwister;
import ESKDB.wdBayesOnlinePYP;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class TestForSplice {

	private static String m_S = "KDB"; // -S (NB,KDB,ESKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static int m_EnsembleSize = 5; // -E
	private static boolean m_Backoff = true;
	private static int m_Tying = 2; // -L
	private static String dataTain;
	private static String dataTest;

	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));
		setOptions(args);

		File sourceFile = new File(dataTain);
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
//		int N = getNumData(sourceFile, structure);

		System.out.println("Dataset : spliceTrain");
//		System.out.println("data size \t" + N);
//		System.out.println("Attribute size \t" + structure.numAttributes());
//		System.out.println("class size \t" + nc);

		double m_RMSE = 0;
		double m_Error = 0;
		int NTest = 0;
		double trainTime = 0;

		wdBayesOnlinePYP learner = new wdBayesOnlinePYP();
		learner.set_m_S(m_S);
		learner.setK(m_K);
		learner.setMEstimation(M_estimation);
		learner.setGibbsIteration(m_IterGibbs);
		learner.setEnsembleSize(m_EnsembleSize);
		learner.setBackoff(m_Backoff);
		learner.setM_Tying(m_Tying);
		learner.setPrint(m_MVerb);

		File trainFile = new File(dataTain);
		long start = System.currentTimeMillis();
		learner.buildClassifier(trainFile);
		trainTime += System.currentTimeMillis() - start;

		
		Instance current;
		File testFile = new File(dataTest);
		reader = new ArffReader(new BufferedReader(new FileReader(testFile), BUFFER_SIZE), 100000);

		start = System.currentTimeMillis();
		while ((current = reader.readInstance(structure)) != null) {
			double[] probs = new double[nc];
			probs = learner.distributionForInstance(current);
			int x_C = (int) current.classValue();

			int pred = -1;
			double bestProb = Double.MIN_VALUE;
			for (int y = 0; y < nc; y++) {
				if (!Double.isNaN(probs[y])) {
					if (probs[y] > bestProb) {
						pred = y;
						bestProb = probs[y];
					}
					m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
				} else {
					System.err.println("probs[ " + y + "] is NaN! oh no!");
				}
			}

			if (pred != x_C) {
				m_Error += 1;
			}
			NTest++;
		}

		m_RMSE = Math.sqrt(m_RMSE / NTest);
		m_Error = m_Error / NTest;

		System.out.println("\n----------------------Bias-Variance Decomposition-------------------");
		System.out.println("Classifier:\t" + m_S);
		System.out.println("Dataset : splice");
		System.out.println("RMSE : " + Utils.doubleToString(m_RMSE, 6, 4));
		System.out.println("Error : " + Utils.doubleToString(m_Error, 6, 4));
		System.out.println("Training time : " + Utils.doubleToString(trainTime, 6, 0));
	}

	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			dataTain = string;
		}

		string = Utils.getOption('T', options);
		if (string.length() != 0) {
			dataTest = string;
		}

		m_MVerb = Utils.getFlag('V', options);
		M_estimation = Utils.getFlag('M', options);

		m_Backoff = Utils.getFlag('B', options);

		string = Utils.getOption('S', options);
		if (string.length() != 0) {
			m_S = string;
		}

		string = Utils.getOption('K', options);
		if (string.length() != 0) {
			m_K = Integer.parseInt(string);
		}

		String ML = Utils.getOption('L', options);
		if (ML.length() != 0) {
			m_Tying = Integer.parseInt(ML);
		}

		string = Utils.getOption('E', options);
		if (string.length() != 0) {
			m_EnsembleSize = Integer.parseInt(string);
		}

		string = Utils.getOption('I', options);
		if (string.length() != 0) {
			m_IterGibbs = Integer.parseInt(string);
		}

		Utils.checkForRemainingOptions(options);
	}

	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			nLines++;
		}
		return nLines;
	}

	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
		File out = File.createTempFile("train-", ".arff");
		out.deleteOnExit();
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);

		Instance current;
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (!testIndexes.get(lineNo)) {
				fileSaver.writeIncremental(current);
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		return out;
	}
}
