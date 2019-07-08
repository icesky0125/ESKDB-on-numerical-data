/*
 * wdBayesOnlinePYP_MDLR Classifier
 * 
 * wdBayesOptMT.java     
 * Code written by:  Francois Petitjean, Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * -V 	Verbosity
 * -S	Structure learning (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis)
 * -M	Parameter learning (M-estimation, HDP)
 * -K 	Value of K for KDB.
 * 
 */
package MDL_R;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import org.apache.commons.math3.util.FastMath;
import ESKDB.xxyDist;
import Method.SmoothingMethod;
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

public final class wdBayesOnlinePYP_MDLR implements Classifier, java.io.Serializable {

	private static final long serialVersionUID = 1L;
	private Instances m_Instances;
	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	private Instances structure;
	public wdBayesParametersTreePYP dParameters_;

	private xxyDist xxyDist_;
	private String m_S = "NB";
	private int m_KDB = 5; // -K
	private boolean m_MVerb = false; // -V
	private BNStructure_MDLR bn = null;
	private static final int BUFFER_SIZE = 100000;
	private static int m_IterGibbs = 50000;
	int m_Tying = 2;

	// added by He Zhang
//	private static boolean M_estimation = false;
	private boolean m_BackOff;
	int[] m_Order;
	int[][] m_Parents;
	int m_BestK_ = 0; 
	int m_BestattIt = 0;
	
	// added by Matthieu Herrmann
	public LogStirlingGenerator lgcache = null;
	
	//added by He Zhang: for HGS smoothing
	protected SmoothingMethod method = SmoothingMethod.HGS;
	private long seed;
	private static boolean M_estimation = false; // default: using HDP
	

	/**
	 * Build Classifier: Reads the source arff file sequentially and build a
	 * classifier. This incorporated learning the Bayesian network structure and
	 * initializing of the Bayes Tree structure to store the count, probabilities,
	 * gradients and parameters.
	 * 
	 * Once BayesTree structure is initialized, it is populated with the counts of
	 * the data.
	 * 
	 * This is followed by discriminative training using SGD.
	 * 
	 * @param sourceFile
	 * @throws Exception
	 */
	public void buildClassifier(File sourceFile) throws Exception {
	
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

		bn = new BNStructure_MDLR(m_Instances, m_S, m_KDB, paramsPerAtt);
		bn.setSeed(seed);
		bn.learnStructure(structure, sourceFile);
		
		

		m_Order = bn.get_Order();
		m_Parents = bn.get_Parents();
		m_BestattIt = bn.get_BestattIt();
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs(); // M-estimation for p(y)
	
		if (this.m_MVerb) {
			System.out.println("************** Display Model **************\n");
			System.out.println("* Attribute order is:\t" + Arrays.toString(this.m_Order));
			for (int u = 0; u < this.nAttributes; u++) {

				System.out.println("parents for attribute " + u + " is:\t" + Arrays.toString(this.m_Parents[u]) + "\t");
			}
			System.out.println();
			
			System.out.println("************** Create Trees Structures for Each Attribute *********\n");
		}

		dParameters_ = new wdBayesParametersTreePYP(m_Instances, paramsPerAtt, m_Order, m_Parents, m_IterGibbs, m_Tying);
		
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
			this.nInstances++;
		}
//		dParameters_.setN(nInstances);
		if (this.m_MVerb) {
			System.out.println("************** Probability Smoothing Started **************");
		}

		switch (method) {
		case M_estimation:
			for (int u = 0; u < this.m_Order.length; u++) {
				ProbabilityTree tree = dParameters_.getPypTrees()[u];
				tree.convertCountToProbs(m_BackOff);
			}

			break;
		case HDP:
			// HDP smoothing
			// sharing one cache for all the trees
			lgcache = LogStirlingFactory.newLogStirlingGenerator(nInstances, 0);

			for (int u = 0; u < this.m_Order.length; u++) {
				ProbabilityTree tree = dParameters_.getPypTrees()[u];
				tree.setLogStirlingCache(lgcache);
				tree.smooth();
				
				if (m_MVerb)
					System.out.println("Tree for attribute " + u + " has been smoothed");
			}
			break;
		case HGS:
			System.out.println("HGS smoothing");
//			ProbabilityTree tree;
//			for (int u = 0; u < this.m_Order.length; u++) {
//				System.out.println("u=="+u);
//				tree = dParameters_.getPypTrees()[u];
//				tree.HGSsmoothing();
//				System.out.println(tree.printFinalPks());
//			}
//			
			break;
		default:
			break;
		}
			
		// used in ESKDB paper
//		if (M_estimation) {
//			for (int u = 0; u < this.m_Order.length; u++) {
//				ProbabilityTree tree = dParameters_.getPypTrees()[u];
//				tree.convertCountToProbs(m_BackOff);
//			}
//		} else {
//			// HDP smoothing
//			// sharing one cache for all the trees
//			lgcache = LogStirlingFactory.newLogStirlingGenerator(nInstances, 0);
//
//			for (int u = 0; u < this.m_Order.length; u++) {
//				ProbabilityTree tree = dParameters_.getPypTrees()[u];
//				tree.setLogStirlingCache(lgcache);
//				tree.smooth();
//				
//				if (m_MVerb)
//					System.out.println("Tree for attribute " + u + " has been smoothed");
//			}
//		}
		
		if (this.m_MVerb) {
			System.out.println("************** Probability Smoothing Finished **************");
		}
		
//		for(int i = 0; i < dParameters_.getPypTrees().length; i++)
//			System.out.println(dParameters_.getPypTrees()[i].printFinalPks());
     	}
	
	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			//probs[c] = FastMath.log(xxyDist_.xyDist_.pp(c));// P(y)
			probs[c] = xxyDist_.xyDist_.pp(c);// P(y)
		}
//		System.out.println(instance);
		for (int u = 0; u < m_BestattIt; u++) {
//			System.out.println(u);
			for (int c = 0; c < nc; c++) {
				double prob = dParameters_.query(instance,u,c);
				probs[c] += FastMath.log(prob);
			}
		}
		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);

		return probs;
	}

	public void setK(int m_K) {
		m_KDB = m_K;
	}

	public void set_m_S(String string) {
		m_S = string;
	}

//	public void setMEstimation(boolean m) {
//		M_estimation = m;
//	}

	public void setTying(int t) {
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
	public double classifyInstance(Instance instance) throws Exception {
		return 0;
	}

	@Override
	public Capabilities getCapabilities() {
		return null;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(data);
		File dataFile = File.createTempFile("train-", ".arff");
		dataFile.deleteOnExit();
		arffSaver.setFile(dataFile);
		arffSaver.writeBatch();

//		this.set_m_S("ESKDB_R");
		buildClassifier(dataFile);
	}

	public void setSmoothingMethod(SmoothingMethod method2) {
		method = method2;
	}

	public void setSeed(long randomSeed) {
		seed = randomSeed;
	}
}
