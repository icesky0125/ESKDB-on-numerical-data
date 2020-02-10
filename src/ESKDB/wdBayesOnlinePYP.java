/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2016 Francois Petitjean, Nayyar A Zaidi and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdBayesOptMT Classifier
 * 
 * wdBayesOptMT.java     
 * Code written by:  Francois Petitjean, Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * -V 	Verbosity
 * -S	Structure learning (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis)
 * -P	Parameter learning (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN, 5: PYP)
 * -K 	Value of K for KDB.
 * 
 */
package ESKDB;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import org.apache.commons.math3.util.FastMath;

import hdp.ProbabilityTree;
import hdp.logStirling.LogStirlingFactory;
import hdp.logStirling.LogStirlingGenerator;
import tools.SUtils;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

public final class wdBayesOnlinePYP implements Classifier,java.io.Serializable {

	private Instances m_Instances;
	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	private Instances structure;

	private xxyDist xxyDist_;
	private String m_S = "NB";
	private int m_KDB = 5; // -K
	private boolean m_MVerb = false; // -V
	private BNStructure bn = null;
	private static final int BUFFER_SIZE = 100000;
	private static int m_IterGibbs = 50000;
	int m_Tying = 2;
	
	// added by He Zhang
	private static boolean M_estimation = false;
	ArrayList<ArrayList<ArrayList<Integer>>> parentOrderforEachAtt;
	private int ensembleSize = 1;
	ArrayList<HashMap<ArrayList<Integer>, ProbabilityTree>> map;
	ArrayList<int[][][]> parentOrder;
	private ArrayList<int[]> upperOrder;
	private boolean m_BackOff;

	// added by Matthieu Herrmann
	public LogStirlingGenerator lgcache = null;
	/**
	 * Build Classifier: Reads the source arff file sequentially and build a
	 * classifier. This incorporated learning the Bayesian network structure and
	 * initializing of the Bayes Tree structure to store the count,
	 * probabilities, gradients and parameters.
	 * 
	 * Once BayesTree structure is initialized, it is populated with the counts
	 * of the data.
	 * 
	 * This is followed by discriminative training using SGD.
	 * 
	 * @param sourceFile
	 * @throws Exception 
	 */
	public void buildClassifier(File sourceFile) throws Exception {
		if(sourceFile == null) {
			System.out.println("stop here");
		}
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		m_Instances = structure;
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		paramsPerAtt = new int[nAttributes + 1];// including class
		for (int u = 0; u < paramsPerAtt.length; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}
		
		bn = new BNStructure(m_Instances, m_S, m_KDB, paramsPerAtt);
		bn.setEnsembleSize(ensembleSize);
		bn.learnStructure(structure, sourceFile);
		
		parentOrderforEachAtt = bn.getParentOrderforEachAtt();
		parentOrder = bn.getLowerOrder();
		upperOrder = bn.getUpperOrder();
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs(); // M-estimation for p(y)
		
		if(this.m_MVerb) {
			System.out.println("************** Display All the SKDBs **************\n");
			
			for (int i = 0; i < parentOrder.size(); i++) {
				System.out.println("* Attribute order for classifier_" + (i+1) + " is:\t"+Arrays.toString(upperOrder.get(i)));
				for (int j = 0; j < parentOrder.get(i).length; j++) {
					for (int z = 0; z < parentOrder.get(i)[j].length; z++) {
						System.out.print("parents for attribute " + upperOrder.get(i)[j] + " is:\t"+Arrays.toString(parentOrder.get(i)[j][z]) + "\t");
					}
					System.out.println();
				}
				System.out.println();
			}
				
			System.out.println("************** Display Parents for Each Attribute ************\n");

			for (int i = 0; i < parentOrderforEachAtt.size(); i++) {
				System.out.print("parents for attribute "+i+" is:\t");
				for (int j = 0; j < parentOrderforEachAtt.get(i).size(); j++) {
						System.out.print(Arrays.toString(parentOrderforEachAtt.get(i).get(j).toArray()) + "\t");
	
				}
				System.out.println();
			}
			System.out.println();
		}
		
		if(this.m_MVerb){
			System.out.println("************** Create Trees Structures for Each Attribute *********\n");
		}
		map = new ArrayList<HashMap<ArrayList<Integer>, ProbabilityTree>>();
		for (int u = 0; u < this.nAttributes; u++) {
			map.add(createTreeMap(u));
		}

		if(this.m_MVerb){
			System.out.println("************** Update Trees With Training Data **************\n");
		}
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			for (int u = 0; u < this.nAttributes; u++) {
				for(ArrayList<Integer> parents: map.get(u).keySet()) {
					ProbabilityTree tempTree  = map.get(u).get(parents);
					int nParents = parents.size();
					int[] datapoint = new int[nParents + 1];
					datapoint[0] = (int) row.value(u);
					for (int p = 0; p < nParents; p++) {
						datapoint[1 + p] = (int) row.value(parents.get(p));
					}
					tempTree.addObservation(datapoint);
				}
			}
			nInstances++;
		}
		
		if(this.m_MVerb){
			System.out.println("************** Probability Smoothing Started **************");
		}
		

		if (M_estimation) {
			for (int u = 0; u < this.nAttributes; u++) {
				for (ProbabilityTree tempTree : map.get(u).values()) {
					tempTree.convertCountToProbs(m_BackOff);
				}
			}
		} else {
			//HDP smoothing 
			
			// sharing one cache for all the trees
			lgcache = LogStirlingFactory.newLogStirlingGenerator(nInstances, 0);

			for (int u = 0; u < this.nAttributes; u++) {
				int count = 0;
				for (ProbabilityTree tempTree : map.get(u).values()) {
					tempTree.setLogStirlingCache(lgcache);
					tempTree.smooth();
					count++;
					if(m_MVerb)
						System.out.println("Tree_" + count + " for attribute " + u +" has been smoothed");
				}
				if(m_MVerb)
					System.out.println();
			}
		}
		
		// System.out.println("\n******************Display trees*************");
//		this.printAllTrees();
	}
	
	public double[] distributionForInstance(Instance instance) {

		this.nc=instance.numClasses();
		double[] probY = new double[nc];
		for (int c = 0; c < nc; c++) {
			probY[c] = xxyDist_.xyDist_.pp(c);// P(y)
		}
		
		ArrayList<Integer> temp;
		int targetNodeValue;
		int[] datapoint;
		int[] parent;
		double[] res = new double[nc];
		
		for (int i = 0; i < upperOrder.size(); i++) {
			// for each upper order
			int[] order = upperOrder.get(i);
			double[] resForOneUpper = Arrays.copyOf(probY, probY.length);
			
			for (int c = 0; c < nc; c++) {
				// for each class
				
				for (int u = 0; u < order.length; u++) {
					
					double tempResForOneParent = 0;
					targetNodeValue = (int) instance.value(order[u]);
					
					// averaged over multiple parent orders
					for (int j = 0; j < parentOrder.get(i)[u].length; j++) {

						parent = parentOrder.get(i)[u][j];
						temp = new ArrayList<Integer>();
						for (int k = 0; k < parent.length; k++) {
							temp.add(parent[k]);
						}
						
						datapoint = new int[parent.length];
						datapoint[0] = c;
						for (int p = 1; p < parent.length; p++) {
							datapoint[p] = (int) instance.value(parent[p]);
						}
						ProbabilityTree tree = map.get(order[u]).get(temp);		
						double a = tree.query(datapoint)[targetNodeValue];
						tempResForOneParent += a;
					}
					tempResForOneParent /= parentOrder.get(i)[u].length;
					double b = FastMath.log(tempResForOneParent);
					resForOneUpper[c] += b;
				}
			}
			
			SUtils.exp(resForOneUpper);
			SUtils.normalize(resForOneUpper);
			for (int c = 0; c < nc; c++) {
				res[c] += resForOneUpper[c];
			}
		}

		for (int c = 0; c < nc; c++) {
			res[c] /= upperOrder.size();
		}
		
		return res;
	}

	public HashMap<ArrayList<Integer>, ProbabilityTree> createTreeMap(int u) {
		ProbabilityTree tree;
		HashMap<ArrayList<Integer>, ProbabilityTree> mapU = new HashMap<ArrayList<Integer>, ProbabilityTree>();

		ArrayList<ArrayList<Integer>> allParentforU = parentOrderforEachAtt.get(u);
		for (int i = 0; i < allParentforU.size(); i++) {
			ArrayList<Integer> parentsU = allParentforU.get(i);

			int nParents = parentsU.size();
			if (!mapU.containsKey(parentsU)) {
				// arityConditioningVariables is values for every parent
				int[] arityConditioningVariables = new int[nParents];
				for (int p = 0; p < nParents; p++) {
					arityConditioningVariables[p] = paramsPerAtt[parentsU.get(p)];
				}
				tree = new ProbabilityTree(paramsPerAtt[u], arityConditioningVariables, m_IterGibbs, m_Tying);
				mapU.put(parentsU, tree);
			}
		}
		return mapU;
	}

	public String getMS() {
		return m_S;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public void setK(int m_K) {
		m_KDB = m_K;
	}

	public void set_m_S(String string) {
		m_S = string;
	}

	public void setM_Iterations(int m) {
		m_IterGibbs = m;
	}

	public void setMEstimation(boolean m) {
		M_estimation = m;
	}

	public void setEnsembleSize(int e) {
		ensembleSize = e;
	}

	public void printAllTrees() {
		for (int i = 0; i < map.size(); i++) {
			for (ProbabilityTree tree : map.get(i).values()) {
				System.out.println(tree.printFinalPks());
			}
		}
	}
	
	public void setM_Tying(int t) {
		m_Tying = t;
	}

	public void setGibbsIteration(int iter) {
		m_IterGibbs = iter;
	}

	public void setBackoff(boolean back) {
		m_BackOff = back;
	}

	public void setPrint(boolean m_MVerb2) {
		this.m_MVerb = m_MVerb2;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		ArffSaver arffSaver = new ArffSaver();
	    arffSaver.setInstances(data);
	    File dataFile = File.createTempFile("train-", ".arff");
	    dataFile.deleteOnExit();
	    arffSaver.setFile(dataFile);
	    arffSaver.writeBatch();
	    
	    this.buildClassifier(dataFile);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		
		return 0;
	}

	@Override
	public Capabilities getCapabilities() {
		return null;
	}
}
