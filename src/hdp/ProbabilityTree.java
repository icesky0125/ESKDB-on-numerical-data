package hdp;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import hdp.logStirling.LogStirlingFactory;
import hdp.logStirling.LogStirlingGenerator;
import hdp.logStirling.LogStirlingGenerator.CacheExtensionException;
import mltools.MathUtils;
import weka.core.Utils;

public class ProbabilityTree {

	private int nIterGibbs;
	private int nBurnIn;
	private int frequencySamplingC;

	LogStirlingGenerator lgCache;
	protected RandomGenerator rng = new MersenneTwister(3071980);
	ProbabilityNode root;
	ArrayList<Concentration> concentrationsToSample;

	ArrayList<HashMap<String, Integer>> valueToIndex;
	ArrayList<ArrayList<String>> indexToValue;

	protected TyingStrategy concentrationTyingStrategy = TyingStrategy.LEVEL;

	int nValuesConditionedVariable;

	int[] nValuesContioningVariables;
	protected int nDatapoints;
	boolean createFullTree = false;
	
	// added for HGS
	ArrayList<ProbabilityNode> leaves = new ArrayList<ProbabilityNode>();
	ArrayList<ProbabilityNode> alphaList = new ArrayList<ProbabilityNode>();
	double precision = 0.00001;
	double step = 0.001;
	double[] bestConcentration = null;
	int NInstance;

	// Constructors

	public ProbabilityTree() {
		init(-1, null, false, 5000, TyingStrategy.LEVEL, 5, false);
	}

	public ProbabilityTree(boolean createFullTree) {
		init(-1, null, createFullTree, 5000, TyingStrategy.LEVEL, 5, false);
	}

	public ProbabilityTree(int m_Iterations, TyingStrategy m_Tying) {
		init(-1, null, false, 5000, m_Tying, 5, false);
	}

	public ProbabilityTree(boolean createFullTree, int m_Iterations, TyingStrategy m_Tying, int frequencySamplingC) {
		init(-1, null, createFullTree, m_Iterations, m_Tying, frequencySamplingC, false);
	}

	public ProbabilityTree(int nValuesConditionedVariable, int[] nValuesConditioningVariables, boolean createFullTree) {
		init(nValuesConditionedVariable, nValuesConditioningVariables, createFullTree, 5000, TyingStrategy.LEVEL, 5,
				true);
	}

	public ProbabilityTree(int nValuesConditionedVariable, int[] nValuesConditioningVariables, int m_Iterations,
			int m_Tying) {
		init(nValuesConditionedVariable, nValuesConditioningVariables, false, m_Iterations,
				this.concentrationTyingStrategy, 5, true);
	}

	public ProbabilityTree(int nValuesConditionedVariable, int[] nValuesConditioningVariables, boolean createFullTree,
			int m_Iterations, TyingStrategy m_Tying, int frequencySamplingC, boolean usePYP, int frequencySamplingD) {
		init(nValuesConditionedVariable, nValuesConditioningVariables, createFullTree, m_Iterations, m_Tying,
				frequencySamplingC, true);
	}

	protected void init(int nValuesConditionedVariable, int[] nValuesConditioningVariables, boolean createFullTree,
			int m_Iterations, TyingStrategy m_Tying, int frequencySamplingC, boolean initRoot) {
		this.nValuesConditionedVariable = nValuesConditionedVariable;
		this.nValuesContioningVariables = nValuesConditioningVariables;
		this.nIterGibbs = m_Iterations;
		setConcentrationTyingStrategy(m_Tying);
		this.nBurnIn = Math.min(1000, nIterGibbs / 10);
		this.frequencySamplingC = frequencySamplingC;
		if (initRoot)
			root = new ProbabilityNode(this, 0, createFullTree);
	}

	public int getNXs() {
		return nValuesContioningVariables.length;
	}

	public double getRisingFact(double x, int n) {
		return FastMath.exp(MathUtils.logGammaRatio(x, n));
	}

	public double digamma(double d) {
		return Gamma.digamma(d);
	}

	protected double logScoreTree() {
		return root.logScoreSubTree();
	}

	/**
	 * Runs the Gibbs sampling for the whole tree with given discount and
	 * concentration parameters
	 * 
	 * @return the log likelihood of the optimized tree
	 */
	public double smooth() {
		// Creating and tying concentrations
		concentrationsToSample = new ArrayList<>();
		switch (concentrationTyingStrategy) {
		case NONE:
			for (int depth = getNXs(); depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ProbabilityNode> nodes = getAllNodesAtDepth(depth);
				for (ProbabilityNode node : nodes) {
					Concentration c = new Concentration();
					concentrationsToSample.add(c);
					node.c = c;
					c.addNode(node);
				}

			}
			break;
		case SAME_PARENT:
			for (int depth = getNXs() - 1; depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ProbabilityNode> nodes = getAllNodesAtDepth(depth);
				for (ProbabilityNode parent : nodes) {
					// creating concentration
					Concentration c = new Concentration();
					concentrationsToSample.add(c);
					for (int child = 0; child < parent.children.length; child++) {
						if (parent.children[child] != null) {
							parent.children[child].c = c;
							c.addNode(parent.children[child]);
						}
					}
				}
			}
			break;
		case LEVEL:
			for (int depth = getNXs(); depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ProbabilityNode> nodes = getAllNodesAtDepth(depth);
				Concentration c = new Concentration();
				concentrationsToSample.add(c);
				for (ProbabilityNode node : nodes) {
					node.c = c;
					c.addNode(node);
				}

			}
			break;
		case SINGLE:
			Concentration c = new Concentration();
			concentrationsToSample.add(c);
			for (int depth = getNXs(); depth > 0; depth--) {
				// tying all children of a node
				ArrayList<ProbabilityNode> nodes = getAllNodesAtDepth(depth);
				for (ProbabilityNode node : nodes) {
					node.c = c;
					c.addNode(node);
				}

			}
			break;
		default:
			break;
		}

		// setting concentration for root
		root.c = new Concentration();
		root.prepareForSamplingTk();

		// Gibbs sampling of the tks, c
		for (int iter = 0; iter < nIterGibbs; iter++) {
			// sample tks once
			for (int depth = getNXs(); depth >= 0; depth--) {
				ArrayList<ProbabilityNode> nodes = getAllNodesAtDepth(depth);
				for (ProbabilityNode node : nodes) {
					node.sampleTks();
				}
			}

			// sample c
			if ((iter + frequencySamplingC / 2) % frequencySamplingC == 0) {
				// sample c once
				for (Concentration c : concentrationsToSample) {
					c.sample(rng);
				}
			}

			if (iter >= nBurnIn) {
				this.recordAndAverageProbabilities();
			}

		}

		double score = logScoreTree();
		return score;
	}

	private ArrayList<ProbabilityNode> getAllNodesAtDepth(int depth) {
		return root.getAllNodesAtRelativeDepth(depth);
	}

	/**
	 * Add the observational data for the leaves Data is stored in a integer format
	 * where each number represents a categorical value from 0 to (nValues - 1)
	 * 
	 * @param data a dataset; first value is the value for the conditioned variable;
	 *             other values are for the conditioning variables (in the order
	 *             given in the constructor)
	 */
	public void addDataset(int[][] data) {
		if (data == null || data.length == 0) {
			throw new RuntimeException("Data is empty");
		}
		int nVariables = data[0].length;
		int nConditioningVariables = nVariables - 1;
		int maxValueConditioned = 0;
		nValuesContioningVariables = new int[nConditioningVariables];

		for (int i = 0; i < data.length; i++) {
			if (data[i][0] > maxValueConditioned) {
				maxValueConditioned = data[i][0];
			}
			for (int j = 1; j < data[i].length; j++) {
				if (data[i][j] > nValuesContioningVariables[j - 1]) {
					nValuesContioningVariables[j - 1] = data[i][j];
				}
			}
		}
		nValuesConditionedVariable = maxValueConditioned + 1;// indexing from 0
		for (int j = 0; j < nValuesContioningVariables.length; j++) {
			nValuesContioningVariables[j]++;
		}
		root = new ProbabilityNode(this, 0, createFullTree);

		for (int[] datapoint : data) {
			root.addObservation(datapoint, 1);
		}

		try {
			lgCache = LogStirlingFactory.newLogStirlingGenerator(data.length, 0.0);
		} catch (NoSuchFieldException | IllegalAccessException e) {
			System.err.println("Log Stirling Cache Exception " + e.getMessage());
			System.err.println("Throws as RuntimeException");
			throw new RuntimeException(e);
		}

		nDatapoints = data.length;
		this.smooth();
	}

	public void addObservation(int[] datapoint) {
		root.addObservation(datapoint, 1);
		nDatapoints++;
	}

	/**
	 * Add the observational data for the leaves Data is stored in a integer format
	 * where each number represents a categorical value from 0 to (nValues - 1)
	 * 
	 * @param data a dataset; first value is the value for the conditioned variable;
	 *             other values are for the conditioning variables (in the order
	 *             given in the constructor)
	 */
	public void addDataset(String[][] data) {
		if (valueToIndex != null) {
			System.out.println("Warning: using existing map of values to index");
		}
		if (data == null || data.length == 0) {
			throw new RuntimeException("Data is empty");
		}

		int nVariables = data[0].length;
		int nConditioningVariables = nVariables - 1;

		// now creating a mapping from String to integer
		valueToIndex = new ArrayList<>(nVariables);
		indexToValue = new ArrayList<>(nVariables);
		for (int i = 0; i < nVariables; i++) {
			valueToIndex.add(new HashMap<String, Integer>());
			indexToValue.add(new ArrayList<String>());
		}

		for (String[] datapoint : data) {
			for (int j = 0; j < datapoint.length; j++) {
				HashMap<String, Integer> map = valueToIndex.get(j);
				String val = datapoint[j];
				if (!map.containsKey(val)) {
					int nValuesForVariable = map.size();
					map.put(val, nValuesForVariable);
					indexToValue.get(j).add(val);
				}
			}
		}
		nValuesConditionedVariable = valueToIndex.get(0).size();
		nValuesContioningVariables = new int[nConditioningVariables];
		for (int j = 0; j < nConditioningVariables; j++) {
			nValuesContioningVariables[j] = valueToIndex.get(j + 1).size();
		}
		root = new ProbabilityNode(this, 0, createFullTree);

		int[] datapointInt = new int[nVariables];
		for (String[] datapoint : data) {
			for (int j = 0; j < datapoint.length; j++) {
				HashMap<String, Integer> map = valueToIndex.get(j);
				datapointInt[j] = map.get(datapoint[j]);
			}
			root.addObservation(datapointInt, 1);
		}

		try {
			lgCache = LogStirlingFactory.newLogStirlingGenerator(data.length, 0.0);
		} catch (NoSuchFieldException | IllegalAccessException e) {
			System.err.println("Log Stirling Cache Exception " + e.getMessage());
			System.err.println("Throws as RuntimeException");
			throw new RuntimeException(e);
		}

		nDatapoints = data.length;
		this.smooth();
	}

	public void smoothTree() {
		if (lgCache == null)
			try {
				lgCache = LogStirlingFactory.newLogStirlingGenerator(nDatapoints, 0.0);
			} catch (NoSuchFieldException | IllegalAccessException e) {
				System.err.println("Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}
		this.smooth();
	}

	public void setLogStirlingCache(LogStirlingGenerator cache) {
		if (lgCache != null) {
			try {
				lgCache.close();
			} catch (Exception e) {
				System.err.println("Closing Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}
		}
		this.lgCache = cache;
	}

	/**
	 * Get the probability estimated by the HDP process
	 * 
	 * @param sample a datapoint (without the target variable)
	 * @return it's probability distribution over the target variable
	 */
	public double[] query(int[] sample) {
		ProbabilityNode node = root;
		for (int n = 0; n < sample.length; n++) {
			if(node.children != null) {
				if (node.children[sample[n]] != null) {
					node = node.children[sample[n]];
				} else {
					break;
				}
			}
		}

		return node.pkAveraged;
	}

	/**
	 * Get the probability estimated by the HDP process
	 * 
	 * @param sample a datapoint (without the target variable)
	 * @return it's probability distribution over the target variable
	 */
	public double[] query(String... sample) {
		ProbabilityNode node = root;
		for (int j = 0; j < sample.length; j++) {
			// +1 because storing the target as well
			int index = valueToIndex.get(j + 1).get(sample[j]);
			if (node.children[index] != null) {
				node = node.children[index];
			} else {
				break;
			}
		}
		return node.pkAveraged;
	}

	public int[] queryMestimation(int[] sample) {
		ProbabilityNode node = root;
		for (int n = 0; n < sample.length; n++) {
			if (node.children[sample[n]] != null) {
				node = node.children[sample[n]];
			} else {
				break;
			}
		}
		return node.nk;
	}

	protected double logStirling(double a, int n, int m) throws CacheExtensionException {

		if (a != lgCache.discountP) {
			try {
				// Do not forget to close to free resources!
				lgCache.close();
			} catch (Exception e) {
				System.err.println("Closing Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}

			try {
				lgCache = LogStirlingFactory.newLogStirlingGenerator(nDatapoints, a);
			} catch (NoSuchFieldException | IllegalAccessException e) {
				System.err.println("Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}
		}

		double res = lgCache.query(n, m);
		return res;

	}

	public String printNks() {
		return root.printNksRecursively("root");
	}

	public String printTks() {
		return root.printTksRecursively("root");
	}

	public String printTksAndNks() {
		return root.printTksAndNksRecursively("root");
	}

	public String printPks() {
		return root.printPksRecursively("root");
	}

	public String printFinalPks() {
		return root.printAccumulatedPksRecursively("root");
	}

	public String printProbabilities() {
		return root.printAccumulatedPksRecursively("root");
	}

	/**
	 * This function samples a dataset from the learned conditional - really this
	 * shouldn't be used unless you have a very specific case
	 * 
	 * @param nDataPoints number of datapoints to generate
	 * @return the generated dataset
	 * @throws NoSuchAlgorithmException
	 */
	public int[][] sampleDataset(int nDataPoints) throws NoSuchAlgorithmException {
		if (nValuesContioningVariables == null) {
			throw new RuntimeException("tree needs to be learnt before sampling a dataset from it");
		}
		int[][] data = new int[nDataPoints][nValuesContioningVariables.length + 1];
		SecureRandom srg = SecureRandom.getInstance("SHA1PRNG");

		for (int i = 0; i < nDataPoints; i++) {

			// choose xs
			ProbabilityNode node = root;
			for (int x = 0; x < nValuesContioningVariables.length; x++) {
				// choose value of x
				int val = srg.nextInt(nValuesContioningVariables[x]);
				data[i][x + 1] = val;
				node = node.children[val];
			}

			// now choosing y given values of xs
			double rand = srg.nextDouble();
			int chosenValue = 0;
			double sumProba = node.pk[chosenValue];
			while (rand > sumProba) {
				chosenValue++;
				assert (chosenValue < node.pk.length);
				sumProba += node.pk[chosenValue];
			}
			data[i][0] = chosenValue;
		}

		return data;
	}

	public void setConcentrationTyingStrategy(TyingStrategy tyingStrategy) {
		this.concentrationTyingStrategy = tyingStrategy;
	}

	public void setConcentrationTyingStrategy(int tyingStrategy) {
		if (tyingStrategy == 0) {
			this.concentrationTyingStrategy = TyingStrategy.NONE;
		} else if (tyingStrategy == 1) {
			this.concentrationTyingStrategy = TyingStrategy.SAME_PARENT;
		} else if (tyingStrategy == 2) {
			this.concentrationTyingStrategy = TyingStrategy.LEVEL;
		} else if (tyingStrategy == 3) {
			this.concentrationTyingStrategy = TyingStrategy.SINGLE;
		}
	}

	public String[] getValuesTarget() {
		String[] values = new String[nValuesConditionedVariable];
		for (int j = 0; j < values.length; j++) {
			values[j] = indexToValue.get(0).get(j);
		}
		return values;
	}

	private void recordAndAverageProbabilities() {
		root.computeProbabilities();
		root.recordAndAverageProbabilities();
	}

	public void convertCountToProbs(boolean m_BackOff) {
		root.convertCountToProbsBackOff(m_BackOff);
	}

	public void setNumInstances(int n) {
		this.NInstance = n;
	}
	
	// ----------------------------------------------------HGS smoothing ------------------------------------------

	public void HGSsmoothing() {
		// 1. get all the internal nodes and the leaves under each internal node
		treeTraversal(root);
		
//		System.out.println("LOO estimated tree:\n"+this.printPks());
		
		// 2. perform gradient descent
		stepGradientLogLoss();
		
		// 3. get the final probabilities at leaves
		double[] sumalpk = new double[this.nValuesConditionedVariable];
		calculatePkForLeavesLogLoss(root, 0, sumalpk, false); 
		// false means calculate final pk, true means calculateLOOCV cost
	
	}
	
	/**
	 * traverse the tree top down to calculate the LOO estimate for internal nodes
	 * nodes
	 */
	public void treeTraversal(ProbabilityNode node) {
		if (node.pk == null) {
			node.pk = new double[this.nValuesConditionedVariable];
		}

		if (node.isLeaf()) {
			leaves.add(node);
			return;
		}

		alphaList.add(node);
		node.leavesUnderThisNode = new ArrayList<ProbabilityNode>();

		int sum = Utils.sum(node.nk);
		for (int i = 0; i < this.nValuesConditionedVariable; i++) {
			
			if (node.nk[i] > 1)
				node.pk[i] = (double) (node.nk[i] - 1) / (sum - 1);
			else {
				node.pk[i] = 0;
			}
		}

		if (node.children != null) {
			for (int c = 0; c < node.children.length; c++) {
				if (node.children[c] != null) {
					treeTraversal(node.children[c]);
					if (node.children[c].isLeaf())
						node.leavesUnderThisNode.add(node.children[c]);
					else {
						node.leavesUnderThisNode.addAll(node.children[c].leavesUnderThisNode);
					}
				}
			}
		}
	}

	//--------------------------------------------------- MSE cost function------------------------------------
	public void stepGradient() {
		double currentCost = LOOCVCost();
		double costDifference = currentCost;
		int iter = 0;
		double newCost = 0;
		while (costDifference > this.precision) {
			// currentCost = this.LOOCVCost();
			ProbabilityNode node;
			for (int i = 0; i < this.alphaList.size(); i++) {
				node = alphaList.get(i);
				node.alpha -= step * node.partialDerivative;
			}
			newCost = this.LOOCVCost();
			costDifference = currentCost - newCost;
			currentCost = newCost;
			// System.out.println(currentCost);
			iter++;
		}
//		System.out.println("\ngradient descend: " + iter + "\t" + newCost);

	}
	
	/**
	 * Minimize the cost function
	 */
	private double LOOCVCost() {

		double[] sumpk = new double[nValuesConditionedVariable];

		// get probability of all the nodes
		calculatePkForLeaves(root, 0, sumpk, true);

		double loocvCost = 0;
		for (int c = 0; c < this.leaves.size(); c++) {
			ProbabilityNode node = leaves.get(c);

			for (int k = 0; k < this.nValuesConditionedVariable; k++) {
				if(node.nk[k] > 1)
					loocvCost += node.nk[k] * Math.pow(1 - node.pk[k], 2);
			}
		}

		loocvCost /= (2 * NInstance);
		return loocvCost;
	}

	//--------------------------------------------------- log loss cost function ------------------------------------
	
	private void stepGradientLogLoss() {
		double currentCost = LOOCVCostLogLoss(); // alphas are all initialized as 2
		double costDifference = currentCost;
		int iter = 0;
		double newCost = 0;
		ProbabilityNode node;

		while (costDifference > this.precision) {

			for (int i = 0; i < this.alphaList.size(); i++) {

				node = this.alphaList.get(i);
				node.alpha -= step * node.partialDerivative;
			}
			newCost = this.LOOCVCostLogLoss();

			costDifference = currentCost - newCost;
			currentCost = newCost;
			iter++;
		}
		System.out.println("iterations needed: "+iter);
	}
	
	/**
	 * Minimize the Log cost function
	 */
	private double LOOCVCostLogLoss() {

		double[] sumpk = new double[nValuesConditionedVariable];
		// get probability of all the nodes
		calculatePkForLeavesLogLoss(root, 0, sumpk, true);

		double loocvCost = 0;
		for (int c = 0; c < this.leaves.size(); c++) {
			ProbabilityNode node = leaves.get(c);

			for (int k = 0; k < this.nValuesConditionedVariable; k++) {
				if(node.nk[k] > 1)
					loocvCost -= node.nk[k] * Math.log(node.pk[k]) / Math.log(2);
			}
		}

		loocvCost /= NInstance;

		return loocvCost;
	}

	public void calculatePkForLeavesLogLoss(ProbabilityNode node, double sumalpha, double[] sumpk, boolean isGD) {

		if (node.isLeaf()) {

			if (isGD) {
				node.alc = new double[this.nValuesConditionedVariable];
				for (int i = 0; i < this.nValuesConditionedVariable; i++) {
					if (node.nk[i] > 0) {
						node.pk[i] = (node.nk[i] - 1 + sumpk[i]) / (Utils.sum(node.nk) - 1 + sumalpha);
						node.alc[i] = node.nk[i] / (Utils.sum(node.nk) - 1 + sumalpha); // belta_l,k
					}
				}
			} else {
				node.pkAveraged = new double[this.nValuesConditionedVariable];
				
				for (int i = 0; i < this.nValuesConditionedVariable; i++) {
					if(Double.isNaN(sumpk[i])) {
						System.out.println("sumpk[i] is nan: "+Arrays.toString(sumpk));
					}
					
					node.pkAveraged[i] = Utils.roundDouble((node.nk[i] + sumpk[i]) / (Utils.sum(node.nk) + sumalpha),4);
					
				}
			}
			return;
		}

		// calculate sumPK and sumAlpha
		sumalpha += node.alpha;
		if (isGD) {
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] += node.pk[i] * node.alpha;
			}

		} else {
			// calculate pk for parents
			int sum = Utils.sum(node.nk);
			node.pkAveraged = new double[this.nValuesConditionedVariable];
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				 node.pkAveraged[i] = tools.SUtils.MEsti(node.nk[i], sum, this.nValuesConditionedVariable); //M-estimation for internal nodes
//				node.pkAveraged[i] = (double) node.nk[i] / sum;
				// node.pkAveraged[i] = (double) (node.nk[i]+1)/(sum+this.nValuesConditionedVariable); // Laplace for internal nodes
			}

			tools.SUtils.normalize(node.pkAveraged);
			// sumPk to calculate the final estimates for leaves
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] += node.pkAveraged[i] * node.alpha;
			}
		}

		// children
		if (node.children != null) {
			for (int s = 0; s < node.children.length; s++) {
				if( node.children[s] != null) {
					calculatePkForLeavesLogLoss(node.children[s], sumalpha, sumpk, isGD);
				}
			}

			sumalpha -= node.alpha;
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] -= node.pk[i] * node.alpha;
			}
		}

		if (isGD) {
			// calculate partial derivative for each internal node
			if (node.leavesUnderThisNode != null) {
				node.partialDerivative = 0;

				for (int c = 0; c < node.leavesUnderThisNode.size(); c++) {
					ProbabilityNode son = node.leavesUnderThisNode.get(c);
					for (int k = 0; k < this.nValuesConditionedVariable; k++) {
						node.partialDerivative += son.alc[k] / son.pk[k] * (son.pk[k] - node.pk[k]);
					}
				}
				node.partialDerivative = 1/(NInstance * Math.log(2));
			}
		}
	}

	public void calculatePkForLeaves(ProbabilityNode node, double sumalpha, double[] sumpk, boolean isGD) {
		if (node.isLeaf()) {
			if (isGD) {
				node.alc = new double[this.nValuesConditionedVariable];
				for (int i = 0; i < this.nValuesConditionedVariable; i++) {
					if (node.nk[i] > 0) {
						node.pk[i] = (node.nk[i] - 1 + sumpk[i]) / (Utils.sum(node.nk) - 1 + sumalpha);
						node.alc[i] = node.nk[i] * (1 - node.pk[i]) / (Utils.sum(node.nk) - 1 + sumalpha);
					}
				}
			} else {
				node.pkAveraged = new double[this.nValuesConditionedVariable];
				for (int i = 0; i < this.nValuesConditionedVariable; i++) {
					node.pkAveraged[i] = (node.nk[i] + sumpk[i]) / (Utils.sum(node.nk) + sumalpha);
				}
			}
			return;
		}

		sumalpha += node.alpha;
		int sum = Utils.sum(node.nk);
		if (isGD) {
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] += node.pk[i] * node.alpha; // pk saves the LOO estimate
			}
		} else {
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				// node.pk[i] = SUtils.MEsti(node.nk[i], sum, this.m_nc);
				node.pk[i] = (double) node.nk[i] / sum;
				// node.pk[i] = (double) (node.nk[i]+1)/(sum+this.m_nc);
			}

			tools.SUtils.normalize(node.pk);

			//
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] += node.pk[i] * node.alpha;
			}
		}

		// first recur on left subtree
		if (node.children != null) {
			for (int s = 0; s < node.children.length; s++) {
				if (node.children[s] != null)
					calculatePkForLeaves(node.children[s], sumalpha, sumpk, isGD);
			}
			sumalpha -= node.alpha;
			for (int i = 0; i < this.nValuesConditionedVariable; i++) {
				sumpk[i] -= node.pk[i] * node.alpha;
			}
		}

		// now deal with the node
		// System.out.print(Arrays.toString(node.nk) + " ");
		if (isGD) {
			if (node.leavesUnderThisNode != null) {
				node.partialDerivative = 0;
				for (int c = 0; c < node.leavesUnderThisNode.size(); c++) {
					ProbabilityNode son = node.leavesUnderThisNode.get(c);
					for (int k = 0; k < this.nValuesConditionedVariable; k++) {
						node.partialDerivative += 2 * son.alc[k] * (son.pk[k] - node.pk[k]);
//						node.partialDerivative += son.alc[k] * (son.pk[k] - node.pk[k]);
					}
				}
			}
			node.partialDerivative /= this.NInstance;

		}
//		else {
//			if(node.pkAveraged == null) {
//				node.pkAveraged  = new double[nValuesConditionedVariable];
//				for (int i = 0; i < this.nValuesConditionedVariable; i++) {
//					node.pkAveraged[i] = (node.nk[i] + sumpk[i]) / (Utils.sum(node.nk) + sumalpha);
//				}
//			}
//		}
	}

	public void prune() {

		this.root.prune();	
	}

	public String printFinalPksForHGS() {
		return root.printAccumulatedPksRecursivelyHGS("root");
	}
}