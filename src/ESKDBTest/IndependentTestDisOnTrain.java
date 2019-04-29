package ESKDBTest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import ESKDB.wdBayesOnlinePYP;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;


public class IndependentTestDisOnTrain {

	private static String m_S = "SKDB"; // -S (NB,KDB,ESKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static int m_EnsembleSize = 20; // -E
	private static boolean m_Backoff = false;
	private static int m_Tying = 2; // -L
	private static String dataTain;
	private static String dataTest;

	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));
		setOptions(args);
		
		File trainFile = new File(dataTain);
		File testFile = new File(dataTest);
		Instances trainData = readFile2Instances(trainFile);
		Instances testData = readFile2Instances(testFile);
		trainData.setClassIndex(trainData.numAttributes()-1);

		int nc = trainData.numClasses();
		String name = trainData.relationName();
		System.out.println("Dataset \t" + name);
		System.out.println("data size \t" + trainData.numInstances());
		System.out.println("Attribute size \t" + trainData.numAttributes());
		System.out.println("class size \t" + nc);

		// discretize on the training set
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(trainData);

		trainData = Filter.useFilter(trainData, disTransform);
		testData = Filter.useFilter(testData, disTransform);
		trainFile = readInstances2File(trainData, "TRAIN");

		// started training
		
		double m_RMSE = 0;
		double m_Error = 0;
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
		
		long start = System.currentTimeMillis();
		learner.buildClassifier(trainFile);
		trainTime += System.currentTimeMillis() - start;
		
		
		// test: each test example is tested using all the discretizers
		Instance current;
		start = System.currentTimeMillis();
		System.out.println("\nStarted testing");
		
		System.out.println("testID:\tProb(0)\tProb(1)\tPrediction\tTrueClass");
		
		int testIndex = 0;
		for( int i = 0; i < testData.numInstances(); i++) {
		
			current = testData.instance(i);
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
			
			String[] probsss = new String[probs.length];
			for(int z = 0; z < probs.length; z++) {
				probsss[z] = Utils.doubleToString(probs[z], 6,3);
			}
			System.out.println("test example "+testIndex +":\t"+probsss[0]+"\t"+probsss[1]+"\t"+pred+"\t"+x_C);
			
			testIndex++;
		}

		m_RMSE = Math.sqrt(m_RMSE / testData.numInstances());
		m_Error = m_Error / testData.numInstances();
		
		double testTime = System.currentTimeMillis() - start;
		
		String smoothing ="";
		if(M_estimation) {
			smoothing = "M_estimation";
		}else {
			smoothing = "HDP";
		}
		System.out.println("\n----------------------Bias-Variance Decomposition-------------------");
		System.out.println("Classifier:\t" + m_S);
		
		System.out.println("Dataset : " + name);
		System.out.println("Smoothing: "+smoothing);
		System.out.println("RMSE : " + Utils.doubleToString(m_RMSE, 6,4));
		System.out.println("Error : " + Utils.doubleToString(m_Error, 6, 4));
		System.out.println("Training time : " + Utils.doubleToString(trainTime, 6, 0));
		System.out.println("Testing time : " + Utils.doubleToString(testTime, 6, 0));
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

	private static Instances readFile2Instances(File trainFile) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(trainFile), BUFFER_SIZE), 10000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances resultInstances = structure;

		Instance row;
		while ((row = reader.readInstance(structure)) != null) {
			resultInstances.add(row);
		}
		return resultInstances;
	}

	private static File readInstances2File(Instances data, String name) throws IOException {
		Instances dataSet = data;
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);

		File res = new File(name + ".arff");
		res.deleteOnExit();
		saver.setFile(res);
		saver.writeBatch();
		return res;
	}
}
