package MemorySolvedESKDBR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeMap;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class OneFoldforSatellite {

	private static String data = "";
	private static String m_S = "KDB"; // -S (NB,KDB,ESKDB)
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

		double m_RMSE = 0;
		double m_Error = 0;
		int NTest = 0;
		double trainTime = 0;
		long randomSeed = 1990093;

		wdBayesOnlinePYP_MDLR learner = new wdBayesOnlinePYP_MDLR();
		learner.set_m_S(m_S);
		learner.setK(m_K);
		learner.setMEstimation(M_estimation);
		learner.setGibbsIteration(m_IterGibbs);
		learner.setBackoff(m_Backoff);
		learner.setTying(m_Tying);
		learner.setPrint(m_MVerb);

		
		File trainFile = new File(dataTain);
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(trainFile), BUFFER_SIZE), 100000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int N = getNumData(trainFile, structure);

		String strData = trainFile.getName().substring(trainFile.getName().lastIndexOf("/") + 1,
				trainFile.getName().lastIndexOf("."));
		System.out.println("Dataset : " + strData);
		System.out.println("data size \t" + N);
		System.out.println("Attribute size \t" + structure.numAttributes());
		System.out.println("class size \t" + nc);
		System.out.println(structure.attribute(structure.classIndex()));
		long start = System.currentTimeMillis();
		wdBayesOnlinePYP_MDLR[] classifiers = new wdBayesOnlinePYP_MDLR[m_EnsembleSize];
		MDLR[] discretizer = new MDLR[m_EnsembleSize];
		
		System.out.println("started learning");

//		// train MDLR and classifier
		for (int k = 0; k < m_EnsembleSize; k++) {

			classifiers[k] = (wdBayesOnlinePYP_MDLR) AbstractClassifier.makeCopy(learner);
			discretizer[k] = classifiers[k].buildClassifier(trainFile, randomSeed);
			System.out.println("The "+k+"th classifier has been trained");
			randomSeed++;
		}

		trainTime += System.currentTimeMillis() - start;

		// test: each test example is tested using all the discretizers
		Instance current;
		File testFile = new File(dataTest);
		reader = new ArffReader(new BufferedReader(new FileReader(testFile), BUFFER_SIZE), 100000);

		System.out.println("started testing");
		
		start = System.currentTimeMillis();
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();// true class label
			double[] probs = new double[nc];

			for (int k = 0; k < discretizer.length; k++) {
				Instance discretizedRow = discretizer[k].discretize(current);

				double[] p = classifiers[k].distributionForInstance(discretizedRow);

				for (int c = 0; c < nc; c++) {
					probs[c] += p[c];
				}
			}

			for (int c = 0; c < nc; c++) {
				probs[c] /= m_EnsembleSize;
			}
			
//			for (int c = 0; c < nc; c++) {
//				System.out.print(Utils.doubleToString(probs[c], 6, 4)+"\t");
//			}
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

//			System.out.println(pred+"\t"+x_C);
			if (pred != x_C) {
				m_Error += 1;
			}
			NTest++;
		}

		m_RMSE = Math.sqrt(m_RMSE / NTest);
		m_Error = m_Error / NTest;

		System.out.println("\n----------------------Bias-Variance Decomposition-------------------");
		System.out.println("Classifier:\t" + m_S);
		System.out.println("Dataset : " + strData);
		System.out.println("RMSE : " + Utils.doubleToString(m_RMSE, 6,4));
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

		while ( reader.readInstance(structure) != null) {
			nLines++;
		}
		return nLines;
	}
}
