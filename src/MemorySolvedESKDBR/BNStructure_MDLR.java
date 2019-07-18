package MemorySolvedESKDBR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import ESKDB.wdBayesParametersTree;
import ESKDB.xxyDist;
import tools.CorrelationMeasures;
import tools.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public final class BNStructure_MDLR {

	private int[][] m_Parents;
	private int[] m_Order;

	int nInstances;
	int nAttributes;
	int nc;
	xxyDist xxyDist_;
	private String m_S = "";
	private int K = 1;
	protected static int MAX_INCORE_N_INSTANCES = 100000;
	private int[] paramsPerAtt;
	private int m_BestK_ = 0;
	private int m_BestattIt = 0;
	
	public BNStructure_MDLR(Instances m_Instances, String m_S, int k, int[] ppa) {
		this.m_S = m_S;
		this.K = k;

		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];

		paramsPerAtt = ppa;

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}

		xxyDist_ = new xxyDist(m_Instances);
//		if (nInstances > 0) {
//			xxyDist_.addToCount(m_Instances);
//		}
	}

	public void learnStructure(File sourceFile, MDLR discretizer,Random generator) throws IOException {

		// First fill xxYDist; everybody needs it
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes()-1);
		
		Instance row;
		while ((row = reader.readInstance(structure)) != null) {
			row = discretizer.discretize(row);
			updateXXYDist(row);
			
			xxyDist_.setNoData(); // N++
			xxyDist_.xyDist_.setNoData(); //N++
		}

		m_BestK_ = K;
		m_BestattIt = nAttributes;
		
//		learnStructureSKDB_R(sourceFile,discretizer,generator);

		switch (m_S) {
		case "NB":
			learnStructureNB();
			break;
		case "TAN":
			learnStructureTAN();
			break;
		case "KDB":
			learnStructureKDB();
			break;
		case "SKDB":
			learnStructureSKDB(structure, sourceFile,discretizer);
			break;
		case "ESKDB":// SKDB with random discretization and sampled attribute orders
			learnStructureSKDB_R(sourceFile,discretizer,generator);
			break;
		default:
			System.out.println("value of m_S has to be in set {NB,TAN,KDB,SKDB,ESKDB_R}");
		}
	}

	private void learnStructureKDB() {

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {

			int nK = Math.min(u, K);

			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				m_ParentsTemp[m_Order[u]] = new int[m_Parents[u].length];

				for (int j = 0; j < m_Parents[u].length; j++) {
					m_ParentsTemp[m_Order[u]][j] = m_Parents[u][j];
				}
			}
		}

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		int[][][] temp = new int[nAttributes][1][];
		for (int i = 0; i < m_Parents.length; i++) {
			temp[i][0] = m_Parents[i];
		}

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}

	}
	
	private void learnStructureSKDB(Instances structure, File sourceFile, MDLR discretizer) throws FileNotFoundException, IOException {
		int m_KDB = m_BestK_;

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nk = Math.min(u, m_KDB);
			if (nk > 0) {
				m_Parents[u] = new int[nk];
				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nk; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// print the structure
		// System.out.println(Arrays.toString(m_Order));
		// for (int i = 0; i < nAttributes; i++) {
		// System.out.print(i + " : ");
		// if (m_Parents[i] != null) {
		// for (int j = 0; j < m_Parents[i].length; j++) {
		// System.out.print(m_Parents[i][j] + ",");
		// }
		// }
		// System.out.println();
		// }

		// System.out.println("**********************************************");
		// System.out.println("SKDB: First Pass Finished");
		// System.out.println("**********************************************");

		wdBayesParametersTree dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order,
				m_Parents, 1);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes()-1);
		Instance instance;
		int N = 0;
		while ((instance = reader.readInstance(structure)) != null) {
			instance = discretizer.discretize(instance);
			dParameters_.update(instance);
			N++;
		}

		dParameters_.countsToProbability();

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Second Pass Finished");
		// System.out.println("**********************************************");

		double[][] foldLossFunctallK_ = new double[m_KDB + 1][nAttributes + 1];
		double[][] posteriorDist = new double[m_KDB + 1][nc];

		/* Start the third costly pass through the data */
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes()-1);
		while ((instance = reader.readInstance(structure)) != null) {
			int x_C = (int) instance.classValue();
			instance = discretizer.discretize(instance);
			for (int y = 0; y < nc; y++) {
				posteriorDist[0][y] = dParameters_.ploocv(y, x_C);
			}
			SUtils.normalize(posteriorDist[0]);

			double error = 1.0 - posteriorDist[0][x_C];
			foldLossFunctallK_[0][nAttributes] += error * error;

			for (int k = 1; k <= m_KDB; k++) {
				for (int y = 0; y < nc; y++) {
					posteriorDist[k][y] = posteriorDist[0][y];
				}
				foldLossFunctallK_[k][nAttributes] += error * error;
			}

			for (int u = 0; u < nAttributes; u++) {
				// Discounting inst from counts
				dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB);

				for (int k = 0; k <= m_KDB; k++)
					SUtils.normalize(posteriorDist[k]);

				for (int k = 0; k <= m_KDB; k++) {
					error = 1.0 - posteriorDist[k][x_C];
					foldLossFunctallK_[k][u] += error * error;
				}
			}
		}

		/* Start the book keeping, select the best k and best attributes */
		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }

		// Proper kdb selective (RMSE)
		for (int k = 0; k <= m_KDB; k++) {
			for (int att = 0; att < nAttributes + 1; att++) {
				foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att] / N);
			}
			// The prior is the same for all values of k_
			foldLossFunctallK_[k][nAttributes] = foldLossFunctallK_[0][nAttributes];
		}

		double globalmin = foldLossFunctallK_[0][nAttributes];

		for (int u = 0; u < nAttributes; u++) {
			for (int k = 0; k <= m_KDB; k++) {
				if (foldLossFunctallK_[k][u] < globalmin) {
					globalmin = foldLossFunctallK_[k][u];
					m_BestattIt = u;
					m_BestK_ = k;
				}
			}
		}

		m_BestattIt += 1;

		if (m_BestattIt > nAttributes)
			m_BestattIt = 0;

		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }
		// System.out.println("globalmin: "+globalmin);
		// System.out.println("Number of features selected is: " + m_BestattIt +
		// " out of " + nAttributes + " features");
		// System.out.println("best k is: " + m_BestK_);

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Third Pass Finished");
		// System.out.println("**********************************************");

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				int nK = Math.min(m_Parents[u].length, m_BestK_);
				m_ParentsTemp[u] = new int[nK];

				for (int j = 0; j < nK; j++) {
					m_ParentsTemp[u][j] = m_Parents[u][j];
				}
			}
		}

		// print the structure
		// System.out.println(Arrays.toString(m_Order));
		// for (int i = 0; i < nAttributes; i++) {
		// System.out.print(i + " : ");
		// if (m_Parents[i] != null) {
		// for (int j = 0; j < m_Parents[i].length; j++) {
		// System.out.print(m_Parents[i][j] + ",");
		// }
		// }
		// System.out.println();
		// }

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		int[] tempAtt = new int[this.m_BestattIt];
		for (int i = 0; i < m_BestattIt; i++) {
			tempAtt[i] = m_Order[i];
		}
		m_Order = tempAtt;
	}

	private void learnStructureNB() {
	}
	
	private void learnStructureTAN() {
		// TAN
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		int[] m_ParentsTemp = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			m_ParentsTemp[u] = -1;
		}
		CorrelationMeasures.findMST(nAttributes, cmi, m_ParentsTemp);

		for (int u = 0; u < nAttributes; u++) {
			if (m_ParentsTemp[u] != -1) {
				m_Parents[u] = new int[1];
				m_Parents[u][0] = m_ParentsTemp[u];
			}
		}

		for (int u = 0; u < nAttributes; u++) {
			m_Order[u] = u;
		}
	}
	
	private void learnStructureSKDB_R(File sourceFile, MDLR discretizer,Random rg) throws FileNotFoundException, IOException {
		
		// the difference between SKDB_R and SKDB is the formor random sample the attribute order for SKDB.
		int m_KDB = m_BestK_;
		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// random sampled attribute order
//		System.out.println(Arrays.toString(mi));
//		int[] orders = Utils.sort(mi);
//		System.out.println(Arrays.toString(orders));
//		Arrays.sort(mi, Collections.reverseOrder());
//		Utils.normalize(mi);
//		System.out.println(Arrays.toString(mi));
		
		m_Order = sampleReNormalizing(m_Order, mi,rg);
		
		int negativeIndex = m_Order.length;
		for(int i = 0; i < m_Order.length; i++) {
			if(m_Order[i] == -1) {
				negativeIndex = i;
				break;
			}
		}
	
		int[] temp = new int[negativeIndex];
		for(int i = 0; i < temp.length; i++) {
			temp[i] = m_Order[i];
		}
		
		
		m_Order = temp;
		m_Parents = new int[negativeIndex][];
		
		// Calculate parents based on MI and CMI
		for (int u = 0; u < m_Order.length; u++) {
			int nk = Math.min(u, m_KDB);
			if (nk > 0) {
				if(m_Order[u] != -1) {
				m_Parents[u] = new int[nk];
				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nk; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
				}
			}
		}
		
		// System.out.println("**********************************************");
		// System.out.println("SKDB: First Pass Finished");
		// System.out.println("**********************************************");

		wdBayesParametersTree dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order,
				m_Parents, 1);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes()-1);
		Instance instance;
		int N = 0;
		while ((instance = reader.readInstance(structure)) != null) {
			instance = discretizer.discretize(instance);
			dParameters_.update(instance);
			N++;
		}

		dParameters_.countsToProbability();

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Second Pass Finished");
		// System.out.println("**********************************************");

		double[][] foldLossFunctallK_ = new double[m_KDB + 1][m_Order.length + 1];
		double[][] posteriorDist = new double[m_KDB + 1][nc];

		/* Start the third costly pass through the data */
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes()-1);
		while ((instance = reader.readInstance(structure)) != null) {
			int x_C = (int) instance.classValue();
			
			instance = discretizer.discretize(instance);

			for (int y = 0; y < nc; y++) {
				posteriorDist[0][y] = dParameters_.ploocv(y, x_C);
			}
			SUtils.normalize(posteriorDist[0]);

			double error = 1.0 - posteriorDist[0][x_C];
			foldLossFunctallK_[0][m_Order.length] += error * error;

			for (int k = 1; k <= m_KDB; k++) {
				for (int y = 0; y < nc; y++) {
					posteriorDist[k][y] = posteriorDist[0][y];
				}
				foldLossFunctallK_[k][m_Order.length] += error * error;
			}

			for (int u = 0; u < m_Order.length; u++) {
				// Discounting inst from counts
				dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB);

				for (int k = 0; k <= m_KDB; k++)
					SUtils.normalize(posteriorDist[k]);

				for (int k = 0; k <= m_KDB; k++) {
					error = 1.0 - posteriorDist[k][x_C];
					foldLossFunctallK_[k][u] += error * error;
				}
			}
			
		}

		/* Start the book keeping, select the best k and best attributes */

		// Proper kdb selective (RMSE)
		for (int k = 0; k <= m_KDB; k++) {
			for (int att = 0; att < m_Order.length + 1; att++) {
				foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att] / N);
			}
			// The prior is the same for all values of k_
			foldLossFunctallK_[k][m_Order.length] = foldLossFunctallK_[0][m_Order.length];
		}

		double globalmin = foldLossFunctallK_[0][m_Order.length];

		for (int u = 0; u < m_Order.length; u++) {
			for (int k = 0; k <= m_KDB; k++) {
				if (foldLossFunctallK_[k][u] < globalmin) {
					globalmin = foldLossFunctallK_[k][u];
					m_BestattIt = u;
					m_BestK_ = k;
				}
			}
		}

		m_BestattIt += 1;

		if (m_BestattIt > m_Order.length)
			m_BestattIt = 0;

		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }
		// System.out.println("globalmin: "+globalmin);
		// System.out.println("Number of features selected is: " + m_BestattIt +
		// " out of " + nAttributes + " features");
		// System.out.println("best k is: " + m_BestK_);

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Third Pass Finished");
		// System.out.println("**********************************************");

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < m_Parents.length; u++) {
			if (m_Parents[u] != null) {
				int nK = Math.min(m_Parents[u].length, m_BestK_);
				m_ParentsTemp[u] = new int[nK];

				for (int j = 0; j < nK; j++) {
					m_ParentsTemp[u][j] = m_Parents[u][j];
				}
			}
		}

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;
	}
	
	private int[] sampleReNormalizing(int[] tempS, double[] tempCMI, Random generator) {
		int[] res = new int[tempCMI.length];
		for(int i = 0; i < res.length; i++) {
			res[i] = -1;
		}

		for (int i = 0; i < tempCMI.length; i++) {

			Utils.normalize(tempCMI);
			double num = generator.nextDouble();
			int index = cumulativeProbability(tempCMI, num);
			res[i] = tempS[index];
			tempCMI[index] = 0;// set the selected probability to be zero, then
								// select another parent again
			if (Utils.sum(tempCMI) == 0) {
				break;
			}
		}
		return res;
	}
	
//	private int[] sampleReNormalizing(int[] tempS, double[] tempCMI) {
//		int[] res = new int[tempCMI.length];
//		for(int i = 0; i < res.length; i++) {
//			res[i] = -1;
//		}
//		long seed = 19900125;
//		for (int i = 0; i < tempCMI.length; i++) {
//
//			Utils.normalize(tempCMI);
////			double p = Math.random();
//			seed = seed + i;
//			Random generator = new Random(seed);
//			double num = generator.nextDouble();
//			int index = cumulativeProbability(tempCMI, num);
//			res[i] = tempS[index];
//			tempCMI[index] = 0;// set the selected probability to be zero, then
//								// select another parent again
//			if (Utils.sum(tempCMI) == 0) {
//				break;
//			}
//		}
//		return res;
//	}
	
	private int cumulativeProbability(double[] array, double p) {

		double cumulativeProbability = 0.0;
		int index = 0;
		for (; index < array.length; index++) {
			cumulativeProbability += array[index];
			if (p <= cumulativeProbability && array[index] != 0) {
				return index;
			}
		}
		return Integer.MAX_VALUE;
	}

	private void updateXXYDist(Instance instance) {
		xxyDist_.update(instance);
	}
	
	public xxyDist get_XXYDist() {
		return xxyDist_;
	}

	public int[] get_Order() {
		return m_Order;
	}

	public int[][] get_Parents() {
		return m_Parents;
	}

	public int get_BestattIt() {
		return m_BestattIt;
	}

	public static boolean isInList(ArrayList<ArrayList<Integer>> list, int[] candidate) {
		
		for (final ArrayList<Integer> item : list) {
			int[] temp = new int[item.size()];
			for(int i = 0; i < temp.length; i++){
				temp[i] = item.get(i).intValue();
			}
			
			if (Arrays.equals(temp, candidate)) {
				return true;
			}
		}
		return false;
	}
}
