package MDL_R;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import Method.SmoothingMethod;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class OneFoldforSplice {

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
	
	protected static SmoothingMethod method = SmoothingMethod.HGS;

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
//		learner.setMEstimation(M_estimation);
		learner.setGibbsIteration(m_IterGibbs);
		learner.setBackoff(m_Backoff);
		learner.setTying(m_Tying);
		learner.setPrint(m_MVerb);
		learner.setSmoothingMethod(method);

		File sourceTrain = new File(dataTain);
		Instances train = readFile2Instances(sourceTrain);
		train.setClassIndex(train.numAttributes() - 1);
		
		File sourceTest = new File(dataTest);
		Instances test = readFile2Instances(sourceTest);

		System.out.println("started to train ESKDB");
		long start = System.currentTimeMillis();
		wdBayesOnlinePYP_MDLR[] classifiers = new wdBayesOnlinePYP_MDLR[m_EnsembleSize];
		
		Instances[] allTests = new Instances[m_EnsembleSize];

		// train MDLR and classifier
		for (int k = 0; k < m_EnsembleSize; k++) {

			MDLR discretizer = new MDLR();
			discretizer.setSeed(randomSeed);
			discretizer.setUseBetterEncoding(true);
			discretizer.setInputFormat(train);
			Instances currentTrain = Filter.useFilter(train, discretizer);
			
			currentTrain.setClassIndex(currentTrain.numAttributes() - 1);
			allTests[k] = Filter.useFilter(test, discretizer);

			classifiers[k] = (wdBayesOnlinePYP_MDLR) AbstractClassifier.makeCopy(learner);
			classifiers[k].setSeed(randomSeed);
			classifiers[k].buildClassifier(currentTrain);
			randomSeed++;
			
			System.out.println("finish training the "+k+"th classifier");
		}

		trainTime += System.currentTimeMillis() - start;
		
		System.out.println("started to test ESKDB");
		
		start = System.currentTimeMillis();

		int testNum = test.numInstances();
		int nc = test.numClasses();
		for (int j = 0; j < testNum; j++) {

			int x_C = (int) test.get(j).classValue();// true class label

			double[] probs = new double[nc];

			for (int k = 0; k < m_EnsembleSize; k++) {

				Instance currentInst = allTests[k].get(j);// discretizatoion
				double[] p = classifiers[k].distributionForInstance(currentInst);

				for (int c = 0; c < nc; c++) {
					probs[c] += p[c];
				}
			}

			for (int c = 0; c < nc; c++) {
				probs[c] /= m_EnsembleSize;
			}
			
			
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
			
			String[] probsss = new String[probs.length];
			for(int z = 0; z < probs.length; z++) {
				probsss[z] = Utils.doubleToString(probs[z], 6,3);
			}
			System.out.println(test.get(j).value(0)+"\t"+probsss[0]+"\t"+probsss[1]+"\t"+pred+"\t"+x_C);

		}

		m_RMSE = Math.sqrt(m_RMSE / NTest);
		m_Error = m_Error / NTest;

		System.out.println("patient"+"\t" + Utils.doubleToString(m_RMSE, 6, 4) + "\t" + Utils.doubleToString(m_Error, 6, 4) + '\t'
				+ trainTime);
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

		string = Utils.getOption('m', options);
		if (string.length() != 0) {
			if(string.equalsIgnoreCase("HGS")){
				method = SmoothingMethod.HGS;
			}else if(string.equalsIgnoreCase("HDP")) {
				method = SmoothingMethod.HDP;
			}else if(string.equalsIgnoreCase("MESTIMATION")) {
				method = SmoothingMethod.M_estimation;
			}else if(string.equalsIgnoreCase("LAPLACE")) {
				method = SmoothingMethod.LAPLACE;
			}
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

	private static File saveInstances2File(Instances data, String name) throws IOException {
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
