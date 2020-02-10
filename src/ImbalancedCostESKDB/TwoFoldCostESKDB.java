package ImbalancedCostESKDB;

/*
 *   This program is 5 times 2-fold cross-validation. 
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.commons.math3.random.MersenneTwister;

import MemorySolvedESKDBR.MDLR;
import MemorySolvedESKDBR.wdBayesOnlinePYP_MDLR;
import Method.SmoothingMethod;
import hdp.logStirling.LogStirlingFactory;
import hdp.logStirling.LogStirlingGenerator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class TwoFoldCostESKDB {

	private static String data = "";
	private static String m_S = "KDB"; // -S (NB,KDB,ESKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	private static int m_nExp = 5;
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static int m_EnsembleSize = 20; // -E
	private static boolean m_Backoff = false;
	private static int m_Tying = 2; // -L
	private static int m_BeginData = 0;
	
	// HDP
	public static LogStirlingGenerator lgcache = null;
	
	//added for unequal cost learning
//	here only test on binary data
	private static int[] m_Cost = {1,2,5,10,100,500}; // here is the cost of predicting positive to negative
	private static double[] total_Cost = new double[m_Cost.length];
	private static double[] m_ErrorCost = new double[m_Cost.length]; // error based on minimum cost
	private static int[] m_classDistribution = {};


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
//		File[] folder = sourceFile.listFiles();
//		Arrays.sort(folder);
//		int size = folder.length;
//		for (int f = m_BeginData; f < size; f++) {
//			sourceFile = folder[f];

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);
			int nc = structure.numClasses();
			int N = getNumData(sourceFile, structure);

			String strData = sourceFile.getName().substring(sourceFile.getName().lastIndexOf("/") + 1,
					sourceFile.getName().lastIndexOf("."));
			
			System.out.print(strData + "\t");
			
//			minIndex is positive, maxIndex is for negative. 
//			Positive has bigger cost than negative.
			int minIndex = Utils.minIndex(m_classDistribution);
			int maxIndex = Utils.maxIndex(m_classDistribution);
		
//			System.out.println("class distribution is "+Arrays.toString(m_classDistribution));
//			System.out.println("positive index is "+minIndex);
//			System.out.println("negative index is "+maxIndex);
//			System.out.println("cost is "+Arrays.toString(m_Cost));
			
			if (!M_estimation) {
				lgcache = LogStirlingFactory.newLogStirlingGenerator(N, 0);
			}

			double m_RMSE = 0;
			double m_Error = 0;
			int NTest = 0;
			long seed = 3071980;
			double trainTime = 0;
			
			long randomSeed = 1990093;

			if (m_MVerb) {
				System.out.println("A 5 times 2-fold cross-validation will be started.");
			}

			/*
			 * Start m_nExp rounds of Experiments
			 */
			for (int exp = 0; exp < m_nExp; exp++) {
				
				
				System.out.print("*");
				if (m_MVerb) {
					System.out.println("-----------------Times " + exp + "----------------------");
				}

				MersenneTwister rg = new MersenneTwister(seed);
				BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);
				// ---------------------------------------------------------
				// Train on Fold 0
				// ---------------------------------------------------------

				wdBayesOnlinePYP_MDLR learner = new wdBayesOnlinePYP_MDLR();
				learner.set_m_S(m_S);
				learner.setK(m_K);
//				learner.setMEstimation(M_estimation);
				if(M_estimation) {
					learner.setSmoothingMethod(SmoothingMethod.M_estimation);
				}else {
					learner.setSmoothingMethod(SmoothingMethod.HDP);
				}
				
				learner.setGibbsIteration(m_IterGibbs);
				learner.setBackoff(m_Backoff);
				learner.setTying(m_Tying);
				learner.setPrint(m_MVerb);

				// creating tempFile for train0
				File trainFile = createTrainTmpFile(sourceFile, structure, test0Indexes);

				long start = System.currentTimeMillis();
				wdBayesOnlinePYP_MDLR[] classifiers = new wdBayesOnlinePYP_MDLR[m_EnsembleSize];
				MDLR[] discretizer = new MDLR[m_EnsembleSize];

				// train MDLR and classifier

				for (int k = 0; k < m_EnsembleSize; k++) {
//					Random generator = new Random(randomSeed);
					classifiers[k] = (wdBayesOnlinePYP_MDLR) AbstractClassifier.makeCopy(learner);

					if (!M_estimation) {
						classifiers[k].setLogStirlingCache(lgcache);
					}
				
					discretizer[k] = classifiers[k].buildClassifier(trainFile,randomSeed);
					randomSeed++;
				}

				trainTime += System.currentTimeMillis() - start;
				if (m_MVerb) {
					System.out.println("Training time fold 1:\t" + (System.currentTimeMillis() - start));
				}

				// ---------------------------------------------------------
				// Test on Fold 1
				// ---------------------------------------------------------

				// test: each test example is tested using all the discretizers	
				int lineNo = 0;
				Instance current;
				int thisNTest = 0;
				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
				
				int FP = 0;
				int FN = 0;
				
				start = System.currentTimeMillis();
				while ((current = reader.readInstance(structure)) != null) {
					if (test0Indexes.get(lineNo)) {
						int x_C = (int) current.classValue();// true class label
						double[] probs = new double[nc];

						for (int k = 0; k < discretizer.length; k++) {
							Instance currentTest = discretizer[k].discretize(current);

							double[] p = classifiers[k].distributionForInstance(currentTest);

							for (int c = 0; c < nc; c++) {
								probs[c] += p[c];
							}
						}

						for (int c = 0; c < nc; c++) {
							probs[c] /= m_EnsembleSize;
						}
						
						

						// ------------------------------------
						// Update RMSE, 0-1 Loss
						// ------------------------------------
						
						int pred = -1; // predicted class by probs
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

						// ------------------------------------
						// Update FP, FN 
						// ------------------------------------
		
						for (int c = 0; c < m_Cost.length; c++) {
							// for each of the cost in the cost array
							
							int predCost = -1; //class prediction made by cost
							
//							if p * costP > (1-p) * costN, return positive, otherwise, return negative
//							minIndex is the index for positive class, maxIndex is for negative
//							here we assume costP is in m_Cost[c], costN is 1.
							if( probs[minIndex] * m_Cost[c] >= probs[maxIndex] * 1 ) {
								predCost = minIndex;
							}else {
								predCost = maxIndex;
							}
							
							if(predCost != x_C) {
								// update FP and FN in order to calculate the totalCost later
								
								if(predCost == minIndex) {
//									predction = P, but actual = N, update FP
									FP++;
								}else {
//									predction = N, but actual = P, update FN
									FN++;		
								}
								
//								update error made by cost
								m_ErrorCost[c] += 1;
							}
						}
						
						thisNTest++;
						NTest++;
					}
					lineNo++;
				}
				
//				System.out.println(FP+"\t"+FN);
				
				//totalCost = costP * FN + 1 * FP 
				for (int c = 0; c < m_Cost.length; c++) {
					total_Cost[c] += m_Cost[c] * FN + 1 * FP;
				}
				

				if (m_MVerb) {
					System.out.println("Testing time fold 1:\t" + (System.currentTimeMillis() - start));
					System.out.println(
							"Testing fold 1 result - RMSE = " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4)
									+ "\t0-1 Loss = " + Utils.doubleToString(m_Error / NTest, 6, 4));

				}

				if (Math.abs(thisNTest - test0Indexes.cardinality()) > 1) {
					System.err.println("no! " + thisNTest + "\t" + test0Indexes.cardinality());
				}

				BitSet test1Indexes = new BitSet(lineNo);
				test1Indexes.set(0, lineNo);
				test1Indexes.xor(test0Indexes);

				// ---------------------------------------------------------
				// Train on Fold 2
				// ---------------------------------------------------------
				learner = new wdBayesOnlinePYP_MDLR();
				learner.set_m_S(m_S);
				learner.setK(m_K);
				learner.setMEstimation(M_estimation);
				learner.setGibbsIteration(m_IterGibbs);
				learner.setBackoff(m_Backoff);
				learner.setTying(m_Tying);
				learner.setPrint(m_MVerb);

				// creating tempFile for train1
				trainFile = createTrainTmpFile(sourceFile, structure, test1Indexes);
				classifiers = new wdBayesOnlinePYP_MDLR[m_EnsembleSize];
				discretizer = new MDLR[m_EnsembleSize];

				start = System.currentTimeMillis();
				for (int k = 0; k < m_EnsembleSize; k++) {
//					Random generator = new Random(randomSeed);
					classifiers[k] = (wdBayesOnlinePYP_MDLR) AbstractClassifier.makeCopy(learner);
					
					if (!M_estimation) {
						classifiers[k].setLogStirlingCache(lgcache);
					}
					
					discretizer[k] = classifiers[k].buildClassifier(trainFile, randomSeed);

					randomSeed++;
				}

				trainTime += System.currentTimeMillis() - start;
				if (m_MVerb) {
					System.out.println("Training time fold 1:\t" + (System.currentTimeMillis() - start));
				}

				// ---------------------------------------------------------
				// Test on Fold 2
				// ---------------------------------------------------------

				thisNTest = 0;
				lineNo = 0;
				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
//				structure = reader.getStructure();
//				structure.setClassIndex(structure.numAttributes()-1);
//				start = System.currentTimeMillis();
				
				FP = 0;
				FN = 0;
				
				while ((current = reader.readInstance(structure)) != null) {
					if (test1Indexes.get(lineNo)) {

						int x_C = (int) current.classValue();// true class label
						double[] probs = new double[nc];

						for (int k = 0; k < discretizer.length; k++) {
							Instance currentTest = discretizer[k].discretize(current);

							double[] p = classifiers[k].distributionForInstance(currentTest);

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
						
						// ------------------------------------
						// Update FP, FN 
						// ------------------------------------
		
						for (int c = 0; c < m_Cost.length; c++) {
							// for each of the cost in the cost array
							
							int predCost = -1; //class prediction made by cost
							
//							if p * costP > (1-p) * costN, return positive, otherwise, return negative
//							minIndex is the index for positive class, maxIndex is for negative
//							here we assume costP is in m_Cost[c], costN is 1.
							if( probs[minIndex] * m_Cost[c] >= probs[maxIndex] * 1 ) {
								predCost = minIndex;
							}else {
								predCost = maxIndex;
							}
							
							if(predCost != x_C) {
								// update FP and FN in order to calculate the totalCost later
								
								if(predCost == minIndex) {
//									predction = P, but actual = N, update FP
									FP++;
								}else {
//									predction = N, but actual = P, update FN
									FN++;		
								}
								
//								update error made by cost
								m_ErrorCost[c] ++;
							}
						}
						
						thisNTest++;
						NTest++;
					}
					lineNo++;
				}
				
//				System.out.println(FP+"\t"+FN);
				
				//totalCost = costP * FN + 1 * FP 
				for (int c = 0; c < m_Cost.length; c++) {
//					for each cost, update the totalcost
					total_Cost[c] += m_Cost[c] * FN + 1 * FP;
				}	
					

				if (m_MVerb) {
					System.out.println("test time fold 2:\t" + (System.currentTimeMillis() - start));
					System.out.println(
							"Testing fold 2 result - RMSE = " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4)
									+ "\t0-1 Loss = " + Utils.doubleToString(m_Error / NTest, 6, 4));
				}

				if (Math.abs(thisNTest - test0Indexes.cardinality()) > 1) {
					System.err.println("no! " + thisNTest + "\t" + test0Indexes.cardinality());
				}

				seed++;
			} // Ends No. of Experiments

			m_RMSE = Math.sqrt(m_RMSE / NTest);
			m_Error = m_Error / NTest;
			trainTime = trainTime / (m_nExp*2);
			
			for ( int c = 0; c < m_Cost.length; c++) {
				total_Cost[c] = total_Cost[c] / NTest;
				m_ErrorCost[c] = m_ErrorCost[c] / NTest;
			}	

//		System.out.println("\n----------------------Bias-Variance Decomposition-------------------");
//		System.out.println("Classifier:\t" + m_S);
//		System.out.println("Dataset : " + strData);
//		System.out.println("RMSE : " + Utils.doubleToString(m_RMSE, 6,4));
//		System.out.println("Error : " + Utils.doubleToString(m_Error, 6, 4));
//		System.out.println("Training time : " + Utils.doubleToString(trainTime, 6, 0));
			System.out.println("\t" + Utils.doubleToString(m_RMSE, 6, 4) +
					"\t" + Utils.doubleToString(m_Error, 6, 4)+ 
					'\t' + Arrays.toString(total_Cost) );
//		}
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

		string = Utils.getOption('X', options);
		if (string.length() != 0) {
			m_nExp = Integer.valueOf(string);
		}

		string = Utils.getOption('I', options);
		if (string.length() != 0) {
			m_IterGibbs = Integer.parseInt(string);
		}

		string = Utils.getOption('R', options);
		if (string.length() != 0) {
			m_BeginData = Integer.parseInt(string);
		}

		Utils.checkForRemainingOptions(options);
	}

	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		Instance ins;
		m_classDistribution = new int[structure.numClasses()];
		while ((ins = reader.readInstance(structure)) != null) {
			nLines++;
			m_classDistribution[(int)ins.value(ins.numAttributes()-1)]++;
			
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
		File out = File.createTempFile(sourceFile + "train-", ".arff");
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
