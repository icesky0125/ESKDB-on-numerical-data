package MemorySolvedESKDBR;

import java.util.Arrays;

import hdp.ProbabilityTree;
import hdp.logStirling.LogStirlingGenerator;
import weka.core.Instance;
import weka.core.Instances;

public class wdBayesParametersTreePYP {

	private ProbabilityTree[] pypTrees;

	private int n; // num of attributes
	private int nc; // num of classes

	private int[] m_ParamsPerAtt;
	private int[] order;
	private int[][] parents;

	private int N;

	// added by Matthieu Herrmann
	public LogStirlingGenerator lgcache = null;

	/**
	 * Constructor called by wdBayes
	 */
	public wdBayesParametersTreePYP(int numC, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents,
			int m_Iterations, int m_Tying) {

		this.n = m_Order.length;
		this.nc = numC;

		m_ParamsPerAtt = new int[n]; // num of values of each attributes
		for (int u = 0; u < n; u++) {
			m_ParamsPerAtt[u] = paramsPerAtt[u];
		}

		order = new int[n];
		parents = new int[n][];

		for (int u = 0; u < m_Order.length; u++) {
			order[u] = m_Order[u];
		}

		for (int u = 0; u < n; u++) {
			if (m_Parents[u] != null) {
				parents[u] = new int[m_Parents[u].length];
				for (int p = 0; p < m_Parents[u].length; p++) {
					parents[u][p] = m_Parents[u][p];
				}
			}
		}

		pypTrees = new ProbabilityTree[n];
		for (int u = 0; u < n; u++) {
			int nParents = (m_Parents[u] == null) ? 0 : m_Parents[u].length;
			int[] arityConditioningVariables = new int[1 + nParents]; // +1 for class
			arityConditioningVariables[0] = nc;
			for (int p = 1; p < arityConditioningVariables.length; p++) {
				arityConditioningVariables[p] = paramsPerAtt[m_Parents[u][p - 1]];
			}

			pypTrees[u] = new ProbabilityTree(paramsPerAtt[m_Order[u]], arityConditioningVariables, m_Iterations,
					m_Tying);
		}
	}

	public void update(Instance instance) {
		for (int u = 0; u < n; u++) {
			ProbabilityTree tree = pypTrees[u];
			// converting instance into pyp lib format
			// +2 bc target and class
			int nParents = (parents[u] == null) ? 0 : parents[u].length;
			int[] datapoint = new int[nParents + 2];
			datapoint[0] = (int) instance.value(order[u]);
			datapoint[1] = (int) (int) instance.value(instance.numAttributes()-1);
//			datapoint[1] = (int) (int) instance.classValue();
			for (int p = 0; p < nParents; p++) {
				datapoint[2 + p] = (int) instance.value(parents[u][p]);
			}
			tree.addObservation(datapoint);
		}
		N++;
	}

//	public void prepareForQuerying() {
//		try {
//			lgcache = LogStirlingFactory.newLogStirlingGenerator(N, 0.0);
//		} catch (NoSuchFieldException e) {
//			e.printStackTrace();
//		} catch (IllegalAccessException e) {
//			e.printStackTrace();
//		}
//		IntStream.range(0, n).parallel().forEach(u -> {
////		IntStream.range(0, n).forEach(u->{
//			ProbabilityTree tree = pypTrees[u];
//			tree.setLogStirlingCache(lgcache);
//			tree.smoothTree();
//			System.out.println("tree " + u + " / " + n + " smoothed");
//		});
//	}

	public double query(Instance instance, int targetNode, int classValue) {
		ProbabilityTree tree = pypTrees[targetNode];
		int nParents = (parents[targetNode] == null) ? 0 : parents[targetNode].length;
		int[] datapoint = new int[nParents + 1];
		datapoint[0] = classValue;
		for (int p = 0; p < nParents; p++) {
			datapoint[1 + p] = (int) instance.value(parents[targetNode][p]);
		}

		int targetNodeValue = (int) instance.value(order[targetNode]);
		double[] prob = tree.query(datapoint);
		
		return prob[targetNodeValue];
	}

	public ProbabilityTree[] getPypTrees() {
		return this.pypTrees;
	}
}
