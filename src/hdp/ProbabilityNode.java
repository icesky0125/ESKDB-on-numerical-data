package hdp;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.random.RandomDataGenerator;

import hdp.logStirling.LogStirlingGenerator.CacheExtensionException;
import mltools.MathUtils;
import tools.SUtils;
import weka.core.Utils;

public class ProbabilityNode {

	/**
	 * Max value for sampling TK
	 */
	public static final int MAX_TK = 1000;

	/**
	 * True count
	 */
	int[] nk;
	/**
	 * sum of nk
	 */
	int marginal_nk;

	/**
	 * Simulated count
	 */
	int[] tk;
	/**
	 * sum of tk
	 */
	int marginal_tk;

	/**
	 * contains the parameters calculated as a function of (c,d,nk,tk)
	 */
	double[] pk;

	/**
	 * contains the accumulated pk for several runs of Gibbs sampling
	 */
	double[] pkAveraged;
	/**
	 * contains the number of pks that have been accumulated in the pkSum
	 */
	int nPkAccumulated;

	Concentration c;

	int varNumberForBanchingChildren;

	public static int windowForSamplingTk = 10;
	double[] probabilityForWindowTk = new double[2 * windowForSamplingTk + 1];

	ProbabilityNode parent;
	ProbabilityNode[] children;
	ProbabilityTree tree;
	
	// added by He Zhang
	private boolean m_BackOff;
	public ArrayList<ProbabilityNode> leavesUnderThisNode;
	public double[] alc;
	public double partialDerivative;
	public double alpha = 2;

	public ProbabilityNode(ProbabilityTree probabilityTree, int varNumberForBanchingChildren) {
		this(probabilityTree, varNumberForBanchingChildren, false);
	}

	public ProbabilityNode(ProbabilityTree probabilityTree, int varNumberForBanchingChildren, boolean createFullTree) {
		this.tree = probabilityTree;
		int nValuesY = tree.nValuesConditionedVariable;
		int[] nValuesXs = tree.nValuesContioningVariables;
		nk = new int[nValuesY];
		tk = new int[nValuesY];
		parent = null;
		this.varNumberForBanchingChildren = varNumberForBanchingChildren;
		if (createFullTree && varNumberForBanchingChildren + 1 <= tree.getNXs()) {
			children = new ProbabilityNode[nValuesXs[varNumberForBanchingChildren]];
			for (int i = 0; i < children.length; i++) {
				children[i] = new ProbabilityNode(this, varNumberForBanchingChildren + 1, createFullTree);
			}
		}
		alpha = 2;
	}

	public ProbabilityNode(ProbabilityNode parent, int varNumberForBanchingChildren) {
		this(parent, varNumberForBanchingChildren, false);
	}

	public ProbabilityNode(ProbabilityNode parent, int varNumberForBanchingChildren, boolean createFullTree) {
		this.parent = parent;
		this.tree = parent.tree;
		int nValuesY = tree.nValuesConditionedVariable;
		int[] nValuesXs = tree.nValuesContioningVariables;
		nk = new int[nValuesY];
		tk = new int[nValuesY];
		this.varNumberForBanchingChildren = varNumberForBanchingChildren;

		if (createFullTree && varNumberForBanchingChildren + 1 <= tree.getNXs()) {
			children = new ProbabilityNode[nValuesXs[varNumberForBanchingChildren]];

			for (int i = 0; i < children.length; i++) {
				children[i] = new ProbabilityNode(this, varNumberForBanchingChildren + 1, createFullTree);
			}
		}

	}

	/**
	 * Add observation to the leaves in the associated subtree
	 * 
	 * @param values      the set of values for the observation; the first is the
	 *                    target (y)
	 * @param xIndexToUse the index of the covariate to use in values; first
	 *                    covariate is at index 1
	 */
	public void addObservation(int[] values, int xIndexToUse) {
		if (isLeaf()) {
			// if at the leaf, then count the data
			nk[values[0]]++;
			marginal_nk++;
		} else {
			nk[values[0]]++;
			marginal_nk++;
			// else just call recursively
			if (children == null) {
				// -1 because values here has y as well
				children = new ProbabilityNode[tree.nValuesContioningVariables[xIndexToUse - 1]];
			}

			if (children[values[xIndexToUse]] == null) {
				children[values[xIndexToUse]] = new ProbabilityNode(this, xIndexToUse);
			}

			children[values[xIndexToUse]].addObservation(values, xIndexToUse + 1);
		}
	}

	public boolean isLeaf() {
		return varNumberForBanchingChildren >= tree.getNXs();
	}

	/**
	 * This function should be called after have seen all the data. - At the leaves,
	 * nk already exist so we just sum create tk respecting the constraints. - At
	 * the intermediate nodes, we first add the tks from the children to form the
	 * current nk and then create the tk
	 */
	public void prepareForSamplingTk() {
		if (children != null) {
			// first we launch the recursive call to make the nk and
			// tk correct for the children
			for (int c = 0; children != null && c < children.length; c++) {
				if (children[c] != null) {
					children[c].prepareForSamplingTk();
				}
			}

			/*
			 * Now the tks (and nks) from the children are correctly set. If a leaf, nk is
			 * already set, so we only have to do it if not a leaf (by summing the tks from
			 * the children).
			 */

			for (int i = 0; i < nk.length; i++) {
				nk[i] = 0;
			}
			marginal_nk = 0;
			for (int c = 0; children != null && c < children.length; c++) {
				if (children[c] != null) {
					for (int k = 0; k < nk.length; k++) {
						int tkChild = children[c].tk[k];
						nk[k] += tkChild;
						marginal_nk += tkChild;
					}
				}
			}
		}

		// Now nks are set for current node; let's initialize the tks

		if (parent == null) {
			for (int k = 0; k < nk.length; k++) {
				tk[k] = (nk[k] == 0) ? 0 : 1;
				marginal_tk += tk[k];
			}
		} else {
			double concentration = getConcentration();
			for (int k = 0; k < nk.length; k++) {
				if (nk[k] <= 1) {
					tk[k] = nk[k];
				} else {
					tk[k] = (int) Math.max(1, Math.floor(
							concentration * (tree.digamma(concentration + nk[k]) - tree.digamma(concentration))));

				}
				marginal_tk += tk[k];
			}
		}
	}

	/**
	 * Computes the log-likelihood function for the tree under the current node
	 * (included)
	 * 
	 * @return
	 */
	public double logScoreSubTree() {
		double res = 0.0;
		res += Concentration.logPochhammerSymbol(c, 0.0, marginal_tk);
		res -= c.logGammaRatioForConcentration(marginal_nk);

		// Now nks are set for current node; let's initialize the tks
		for (int k = 0; k < nk.length; k++) {

			try {
				res += tree.logStirling(0.0, nk[k], tk[k]);
			} catch (CacheExtensionException e) {
				System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
				e.printStackTrace();
				System.exit(1);
			}

			if (res == Double.NEGATIVE_INFINITY) {
				throw new RuntimeException("log stirling return neg infty");
			}
		}
		// we score all of the children (doesn't matter if done first or after)
		for (int c = 0; children != null && c < children.length; c++) {
			if (children[c] != null) {
				res += children[c].logScoreSubTree();
			}
		}

		return res;

	}

	public String printNksRecursively(String prefix) {
		String res = "";

		// root node
		res += prefix + ":nk=" + Arrays.toString(nk) + "\n";
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if(children[c] != null)
					res += children[c].printNksRecursively(prefix + " -> " + c);
			}
		}
		return res;
	}

	public String printTksRecursively(String prefix) {
		String res = "";

		// root node
		res += prefix + ":tk=" + Arrays.toString(tk) + "\n";
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				res += children[c].printTksRecursively(prefix + " -> " + c);
			}
		}
		return res;
	}

	public String printTksAndNksRecursively(String prefix) {
		String res = "";

		// root node
		res += prefix + ":tk=" + Arrays.toString(tk) + " :nk=" + Arrays.toString(nk) + " :c=" + this.c + "\n";
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					res += children[c].printTksAndNksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printPksRecursively(String prefix) {
		String res = "";

		res += prefix + ":pk=" + Arrays.toString(pk) + " c=" + this.c + "\n";
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					res += children[c].printPksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printAccumulatedPksRecursively(String prefix) {
		String res = "";

		// root node
		res += prefix + ":pk=" + Arrays.toString(pkAveraged) + " c=" + this.c + "\n";
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					res += children[c].printAccumulatedPksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public ArrayList<ProbabilityNode> getAllNodesAtRelativeDepth(int depth) {
		ArrayList<ProbabilityNode> res = new ArrayList<>();
		if (depth == 0) {
			res.add(this);
		} else {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					res.addAll(children[c].getAllNodesAtRelativeDepth(depth - 1));
				}
			}
		}
		return res;
	}

	/**
	 * add a value to the current one of the tk[k]
	 * 
	 * @param k   the index of the value to change in tk
	 * @param the value to set tk
	 * @return the non-normalized posterior probability at this point;
	 *         negative-infinity if value not authorized
	 */
	protected double setTk(int k, int val) {
		// how much to increment (or decrement tk by)
		int incVal = val - tk[k];
		if (incVal < 0) {
			// if decrement, then have to check that valid for the
			// parent
			if (parent != null && incVal < 0 && (parent.nk[k] + incVal) < parent.tk[k]) {
				// not valid; skip
				return Double.NEGATIVE_INFINITY;
			}
		}

		tk[k] += incVal;
		marginal_tk += incVal;

		double res = 0.0;

		// partial score difference for current node
		try {
			res += tree.logStirling(0.0, nk[k], tk[k]);
		} catch (CacheExtensionException e) {
			System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
			e.printStackTrace();
			System.exit(1);
		}

		res += Concentration.logPochhammerSymbol(c, 0.0, marginal_tk);

		// partial score difference for parent
		if (parent != null) {
			parent.nk[k] += incVal;
			parent.marginal_nk += incVal;

			try {
				res += tree.logStirling(0.0, parent.nk[k], parent.tk[k]);
			} catch (CacheExtensionException e) {
				System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
				e.printStackTrace();
				System.exit(1);
			}

			res -= parent.c.logGammaRatioForConcentration(parent.marginal_nk);
		}

		return res;

	}

	public void sampleTks() {
		if (parent == null) {
			// case for root: no sampling, t is either 0 or 1
			for (int k = 0; k < tk.length; k++) {
				// Wray says this is GEM
				int t = (nk[k] == 0) ? 0 : 1;
				setTk(k, t);
			}
		} else {
			for (int k = 0; k < tk.length; k++) {
				if (nk[k] <= 1) {
					/*
					 * can't sample anything, constraints say that tk[k] must be nk[k] just have to
					 * check that tk[k] is different or not to the previous time (in case nk[k] has
					 * just changed)
					 */
					setTk(k, nk[k]);
				} else {
					// sample case
					// starting point
					int oldTk = tk[k];
					int valTk = tk[k] - windowForSamplingTk;
					// maxTk can't be larger than nk[k]
					int maxTk = Math.min(tk[k] + windowForSamplingTk, nk[k]);

					// Limit maxTk for big dataset
					if (maxTk > MAX_TK) {
						maxTk = MAX_TK;
					}

					int index = 0;
					while (valTk < 1) {// move to first allowed position
						probabilityForWindowTk[index] = Double.NEGATIVE_INFINITY;
						valTk++;
						index++;
					}
					boolean hasOneValue = false;
					while (valTk <= maxTk) {// now fill posterior
						double logProbDifference = setTk(k, valTk);
						probabilityForWindowTk[index] = logProbDifference;
						hasOneValue = (hasOneValue || probabilityForWindowTk[index] != Double.NEGATIVE_INFINITY);
						index++;
						valTk++;
					}
					if (!hasOneValue) {
						setTk(k, oldTk);
						continue;
					}
					for (; index < probabilityForWindowTk.length; index++) {
						// finish filling with neg infty
						probabilityForWindowTk[index] = Double.NEGATIVE_INFINITY;
					}

					// now lognormalize probabilityForWindowTk and exponentiate
					MathUtils.normalizeInLogDomain(probabilityForWindowTk);
					MathUtils.exp(probabilityForWindowTk);

					for (int j = 0; j < probabilityForWindowTk.length; j++) {
						if (Double.isNaN(probabilityForWindowTk[j])) {
							System.err.println("problem " + Arrays.toString(probabilityForWindowTk));
						}
					}
					// now sampling tk according to probability vector
					int chosenIndex = MathUtils.sampleFromMultinomial(tree.rng, probabilityForWindowTk);

					// assign chosen tk
					int valueTkChosen = oldTk - windowForSamplingTk + chosenIndex;
					setTk(k, valueTkChosen);
				}
			}
		}
	}

	public double getConcentration() {
		if (c == null) {
			return 2.0;
		} else {
			return c.getConcentration();
		}
	}

	public boolean checkNkSumTks() {
		if (children != null) {
			for (int k = 0; k < nk.length; k++) {
				int sumTkChildren = 0;
				for (int c = 0; c < children.length; c++) {
					if (children[c] != null) {
						sumTkChildren += children[c].tk[k];
					}
				}

				if (sumTkChildren != nk[k]) {
					System.out.println(sumTkChildren + " != " + nk[k]);
					return false;
				}
			}
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null && !children[c].checkNkSumTks()) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * To be called to free the memory of the local caches
	 */
	public void clearMemoryAfterSmoothing() {
		pk = null;
		tk = null;
		probabilityForWindowTk = null;
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					children[c].clearMemoryAfterSmoothing();
				}
			}
		}
	}

	public void createSyntheticSubTree(RandomDataGenerator rdg) {
		if (pk == null) {
			pk = new double[tk.length];
		}
		// sample some concentration
		double parentConcentration = (parent == null) ? 2.0 : parent.getConcentration();
		double[] parentProbs;
		if (parent == null) {
			// root case
			parentProbs = new double[pk.length];
			Arrays.fill(parentProbs, 1.0 / tree.nValuesConditionedVariable);
		} else {
			// normal case
			parentProbs = parent.pk;
		}

		// now sampling pk from parentPk and concentration
		double sumPk = 0.0;
		for (int k = 0; k < pk.length; k++) {
			pk[k] = rdg.nextGamma(Math.max(parentProbs[k] * parentConcentration, 1e-75), 1.0);
			sumPk += pk[k];
		}
		for (int k = 0; k < pk.length; k++) {
			pk[k] /= sumPk;
		}

		if (children != null) {
			// choose concentration for this (to be used by children)
			c = new Concentration();
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					children[c].c = this.c;
					this.c.addNode(children[c]);
					children[c].createSyntheticSubTree(rdg);
				}
			}
			c.sample(rdg.getRandomGenerator());
		}

	}

	public int getNOutcomesTarget() {
		return tree.nValuesConditionedVariable;
	}

	// --- --- --- MEstimation

	public void convertCountToProbs() {
		if (isLeaf()) {
			// if at the leaf, then count to probabilities
			pkAveraged = new double[nk.length];
			for (int i = 0; i < nk.length; i++) {
				pkAveraged[i] = MathUtils.MEsti(nk[i], marginal_nk, nk.length);
			}
		} else {
			// else just call recursively
			if (children != null) {
				for (int i = 0; i < children.length; i++) {
					if (children[i] != null) {
						children[i].convertCountToProbs();
					}
				}
			}
		}
	}

	public void convertCountToProbsBackOff(boolean m_BackOff) {
		pkAveraged = new double[nk.length];
		if (MathUtils.sum(nk) != 0) {
			for (int i = 0; i < nk.length; i++) {
				pkAveraged[i] = Utils.roundDouble(SUtils.MEsti(nk[i], marginal_nk, nk.length),4);
			}
		} else {
			if (m_BackOff) {
				// Here, this.parent is never null because sum(nk) is never 0 for the root
				pkAveraged = this.parent.pkAveraged;
			} else {// uniform
				for (int i = 0; i < nk.length; i++) {
					pkAveraged[i] = (double) 1 / nk.length;
				}
			}
		}

		if (children != null) {
			for (int i = 0; i < children.length; i++) {
				if (children[i] != null) {
					children[i].convertCountToProbsBackOff(m_BackOff);
				}
			}
		}
	}

	// --- --- --- END OF MEstimation

	/**
	 * This function computes the values of the smoothed conditional probabilities
	 * as a function of (nk,tk,c,d) and of the parent probability. <br/>
	 * p_k = ( ( nk - tk*d ) / (N + c) ) ) + ( ( c + T*d ) / (N + c) ) ) *
	 * p^{parent}_k
	 * 
	 * @see <a href=
	 *      "http://topicmodels.org/2014/11/13/training-a-pitman-yor-process-tree-with-observed-data-at-the-leaves-part-2/">
	 *      topicmodels.org</a> (Equation 1)
	 */
	public void computeProbabilities() {

		if (pk == null) {
			pk = new double[nk.length];
		}
		double concentration = getConcentration();
		double sum = 0.0;
		for (int k = 0; k < pk.length; k++) {
			double parentProb = (this.parent != null) ? this.parent.pk[k] : 1.0 / pk.length;// uniform parent if root
																							// node

			pk[k] = (nk[k]) / (marginal_nk + concentration)
					+ (concentration) * parentProb / (marginal_nk + concentration);
			sum += pk[k];
		}

		// normalize
		for (int k = 0; k < pk.length; k++) {
			pk[k] /= sum;
		}

		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					children[c].computeProbabilities();
				}
			}
		}
	}

	/**
	 * This method accumulates the pks so that the final result is averaged over
	 * several successive iterations of the Gibbs sampling process in log space to
	 * avoid underflow
	 */
	protected void recordAndAverageProbabilities() {
		// in this method, pkAveraged stores the log sum
		if (this.pkAveraged == null) {
			pkAveraged = new double[nk.length];
			nPkAccumulated = 1;
		}

		double sum = 0.0;
		for (int k = 0; k < pkAveraged.length; k++) {
			pkAveraged[k] += (pk[k] - pkAveraged[k]) / nPkAccumulated;
			sum += pkAveraged[k];
		}
		// normalize
		for (int k = 0; k < pk.length; k++) {
			pkAveraged[k] /= sum;
		}
		nPkAccumulated++;

		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					children[c].recordAndAverageProbabilities();
				}
			}
		}
	}

	public void prune() {
		boolean pure = false;
		
		// check if the current node is pure
		if(!this.isLeaf()) {
			for(int i = 0; i < this.nk.length; i++) {
				if(this.nk[i] == this.marginal_nk) {
					if(this.children != null) {
						this.children = null;
					}
					System.out.println("remove the children because the current node is pure");
					pure = true;
					break;
				}
			}
		}
		
		// check if the children are pure or not.
		if(!pure) {
			if(this.children != null) {
				for(int c = 0; c < children.length; c++) {
					if(children[c] != null)
						children[c].prune();
				}
			}
		}	
	}

	public String printAccumulatedPksRecursivelyHGS(String prefix) {
		String res = "";

		// root node
		if(!this.isLeaf())
			res += prefix + ":pk=" + Arrays.toString(pkAveraged) + " alpha=" + this.alpha + "\n";
		else {
			res += prefix + ":pk=" + Arrays.toString(pkAveraged) + "\n";
		}
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				if (children[c] != null) {
					res += children[c].printAccumulatedPksRecursivelyHGS(prefix + " -> " + c);
				}
			}
		}
		return res;
	}
	
}
