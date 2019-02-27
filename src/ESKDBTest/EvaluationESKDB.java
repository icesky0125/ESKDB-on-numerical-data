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
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;

public class EvaluationESKDB {

	private static String data = "";
	private static String m_S = "KDB"; // -S (NB,KDB,ESKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static int m_EnsembleSize = 5; // -E 
	private static boolean m_Backoff = true;
	private static int m_Tying = 2; // -L 
	
	
	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));
		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File source;
		source = new File(data);
		if (!source.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

		File sourceFile = new File(data);
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int N = getNumData(sourceFile, structure);
		
		double m_RMSE = 0;
		double m_Error = 0;
		int NTest = 0;
		int NTrain = 0;
		long seed = 3071980;
		double trainTime = 0;
		double testTime = 0;
		
		
		MersenneTwister rg = new MersenneTwister(seed);
		BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);
	
		wdBayesOnlinePYP learner = new wdBayesOnlinePYP();
		learner.set_m_S(m_S);
		learner.setK(m_K);
//		learner.setRandomGenerator(rg);
		learner.setMEstimation(M_estimation);
		learner.setGibbsIteration(m_IterGibbs);
		learner.setEnsembleSize(m_EnsembleSize);
		learner.setBackoff(m_Backoff);
		learner.setM_Tying(m_Tying);
		learner.setPrint(m_MVerb);

		// creating training data
		File trainFile = createTrainTmpFile(sourceFile, structure, test0Indexes);
		NTrain = getNumData(trainFile, structure);
		
		if(m_MVerb) {
			System.out.println("\n---------------------- Training Started ----------------------");
		}
		
		long start = System.currentTimeMillis();
		learner.buildClassifier(trainFile);
		trainTime = System.currentTimeMillis() - start;
		
		if(m_MVerb) {
			System.out.println("\n--------------------- Training Finished ---------------------");
		}
		int lineNo = 0;
		Instance current;
	
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);

		if(m_MVerb) {
			System.out.println("\n---------------------- Testing Started ----------------------");
		}
		start = System.currentTimeMillis();
		while ((current = reader.readInstance(structure)) != null) {
			if (test0Indexes.get(lineNo)) {
				double[] probs = new double[nc];
				probs = learner.distributionForInstance(current);
				int x_C = (int) current.classValue();

				// ------------------------------------
				// Update Error and RMSE
				// ------------------------------------
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
			lineNo++;
		}

		testTime = System.currentTimeMillis()-start;
		if(m_MVerb) {
			System.out.println("\n--------------------- Testing Finished ----------------------");
		}
		
		System.out.println("\n--------------------- Data Information ----------------------");
		String strData = data.substring(data.lastIndexOf("/") + 1, data.lastIndexOf("."));
		System.out.println("Dataset :\t\t" + strData);
		System.out.println("Attribute size :\t" + structure.numAttributes());
		System.out.println("class size :\t\t" + nc);
		System.out.println("data size :\t\t" + N);
		System.out.println("Training data size :\t"+ NTrain);
		System.out.println("Testing data size :\t"+ NTest);

		
		System.out.println("\n---------------- Bias-Variance Decomposition ----------------");
		System.out.println("Classifier:\t" + m_S);
		System.out.println("Training time :\t" + Utils.doubleToString(trainTime, 6, 0));
		System.out.println("Testing time :\t" + Utils.doubleToString(testTime, 6, 0));
		System.out.println("RMSE :\t\t" + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
		System.out.println("Error :\t\t" + Utils.doubleToString(m_Error / NTest, 6, 4));
	}
	
	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
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

	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg)
			throws FileNotFoundException, IOException {
		BitSet res = new BitSet();
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if (rg.nextBoolean()) {
				res.set(nLines);
			}
			nLines++;
		}

		int expectedNLines = (nLines % 2 == 0) ? nLines / 2 : nLines / 2 + 1;
		int actualNLines = res.cardinality();

		if (actualNLines < expectedNLines) {
			while (actualNLines < expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (res.get(chosen));
				res.set(chosen);
				actualNLines++;
			}
		} else if (actualNLines > expectedNLines) {
			while (actualNLines > expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (!res.get(chosen));
				res.clear(chosen);
				actualNLines--;
			}
		}
		return res;
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
