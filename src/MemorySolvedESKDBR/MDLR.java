package MemorySolvedESKDBR;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import tools.SUtils;
import weka.core.*;
import weka.filters.supervised.attribute.Discretize;;

public class MDLR extends Discretize {

	/** for serialization */
	static final long serialVersionUID = -3141006402280129097L;

	public MDLR() {
		super.setAttributeIndices("first-last");
	}

	/**
	 * Signifies that this batch of input to the filter is finished. If the filter
	 * requires all instances prior to filtering, output() may now be called to
	 * retrieve the filtered instances.
	 * 
	 * @return true if there are instances pending output
	 * @throws IllegalStateException if no input structure has been defined
	 */
	@Override
	public boolean batchFinished() {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		if (m_CutPoints == null) {
			calculateCutPoints();

			setOutputFormat();

			// If we implement saving cutfiles, save the cuts here

			// Convert pending input instances
			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				convertInstance(getInputFormat().instance(i));
			}
		}
		flushInput();

		m_NewBatch = true;
		return (numPendingOutput() != 0);
	}

	/** Generate the cutpoints for each attribute */
	protected void calculateCutPoints() {

		Instances copy = null;

		m_CutPoints = new double[getInputFormat().numAttributes()][];
		for (int i = getInputFormat().numAttributes() - 1; i >= 0; i--) {
			if ((m_DiscretizeCols.isInRange(i)) && (getInputFormat().attribute(i).isNumeric())) {

				// Use copy to preserve order
				if (copy == null) {
					copy = new Instances(getInputFormat());
				}
				calculateCutPointsByMDL(i, copy);
			}
		}
	}

	/**
	 * Set cutpoints for a single attribute using MDL.
	 * 
	 * @param index the index of the attribute to set cutpoints for
	 * @param data  the data to work with
	 */
	protected void calculateCutPointsByMDL(int index, Instances data) {

		// Sort instances
		data.sort(data.attribute(index));

		// Find first instances that's missing
		int firstMissing = data.numInstances();
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).isMissing(index)) {
				firstMissing = i;
				break;
			}
		}
		m_CutPoints[index] = cutPointsForSubset(data, index, 0, firstMissing, true);
	}

	/**
	 * Test using Kononenko's MDL criterion.
	 * 
	 * @param priorCounts
	 * @param bestCounts
	 * @param numInstances
	 * @param numCutPoints
	 * @return true if the split is acceptable
	 */
	private double KononenkosMDL(double[] priorCounts, double[][] bestCounts, double numInstances, int numCutPoints) {

		double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
		double before, after;
		int numClassesTotal;

		// Number of classes occuring in the set
		numClassesTotal = 0;
		for (double priorCount : priorCounts) {
			if (priorCount > 0) {
				numClassesTotal++;
			}
		}

		// Encode distribution prior to split
		distPrior = SpecialFunctions.log2Binomial(numInstances + numClassesTotal - 1, numClassesTotal - 1);

		// Encode instances prior to split.
		instPrior = SpecialFunctions.log2Multinomial(numInstances, priorCounts);

		before = instPrior + distPrior;

		// Encode distributions and instances after split.
		for (double[] bestCount : bestCounts) {
			sum = Utils.sum(bestCount);
			distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1, numClassesTotal - 1);
			instAfter += SpecialFunctions.log2Multinomial(sum, bestCount);
		}

		// Coding cost after split
		after = Utils.log2(numCutPoints) + distAfter + instAfter;

		// Check if split is to be accepted
		return (before - after);
	}

	private double calculateInformationGain(double[] priorCounts, double[][] bestCounts) {
		double priorEntropy, entropy, gain;
		// Compute entropy before split.
		priorEntropy = ContingencyTables.entropy(priorCounts);

		// Compute entropy after split.
		entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

		// Compute information gain.
		gain = priorEntropy - entropy;
		return gain;

	}

	/**
	 * Test using Fayyad and Irani's MDL criterion.
	 * 
	 * @param priorCounts
	 * @param bestCounts
	 * @param numInstances
	 * @param numCutPoints
	 * @return true if the splits is acceptable
	 */
	private double FayyadAndIranisMDL(double[] priorCounts, double[][] bestCounts, double numInstances,
			int numCutPoints) {

		double priorEntropy, entropy, gain;
		double entropyLeft, entropyRight, delta;
		int numClassesTotal, numClassesRight, numClassesLeft;

		// Compute entropy before split.
		priorEntropy = ContingencyTables.entropy(priorCounts);

		// Compute entropy after split.
		entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

		// Compute information gain.
		gain = priorEntropy - entropy;

		// Number of classes occuring in the set
		numClassesTotal = 0;
		for (double priorCount : priorCounts) {
			if (priorCount > 0) {
				numClassesTotal++;
			}
		}

		// Number of classes occuring in the left subset
		numClassesLeft = 0;
		for (int i = 0; i < bestCounts[0].length; i++) {
			if (bestCounts[0][i] > 0) {
				numClassesLeft++;
			}
		}

		// Number of classes occuring in the right subset
		numClassesRight = 0;
		for (int i = 0; i < bestCounts[1].length; i++) {
			if (bestCounts[1][i] > 0) {
				numClassesRight++;
			}
		}

		// Entropy of the left and the right subsets
		entropyLeft = ContingencyTables.entropy(bestCounts[0]);
		entropyRight = ContingencyTables.entropy(bestCounts[1]);

		// Compute terms for MDL formula
		delta = Utils.log2(Math.pow(3, numClassesTotal) - 2) - ((numClassesTotal * priorEntropy)
				- (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));

		// Check if split is to be accepted
		return (gain - (Utils.log2(numCutPoints) + delta) / numInstances);
	}
	
	/**
	 * random sample for cut points from the IG distribution. added by He Zhang.
	 * 
	 * @param instances
	 * @param attIndex
	 * @param first
	 * @param lastPlusOne
	 * @param firstFlag
	 * @return selected cut points
	 */

	private double[] cutPointsForSubset(Instances instances, int attIndex, int first, int lastPlusOne,
			boolean firstFlag) {

		double[][] counts, bestCounts;
		double[] priorCounts, left = null, right = null, cutPoints;
		double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
		int bestIndex = -1, numCutPoints = 0;
		double numInstances = 0;

		// Compute number of instances in set
		if ((lastPlusOne - first) < 2) {
			return null;
		}

		// Compute class counts.
		counts = new double[2][instances.numClasses()];
		for (int i = first; i < lastPlusOne; i++) {
			numInstances += instances.instance(i).weight();
			counts[1][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
		}

		// Save prior counts
		priorCounts = new double[instances.numClasses()];
		System.arraycopy(counts[1], 0, priorCounts, 0, instances.numClasses());

		// Entropy of the full set
		priorEntropy = ContingencyTables.entropy(priorCounts);
		bestEntropy = priorEntropy;

		// Use worse encoding?
		if (!m_UseBetterEncoding) {
			numCutPoints = (lastPlusOne - first) - 1;
		}

		// find all possible cut points and partitions
		ArrayList<Integer> index = new ArrayList<Integer>();
		ArrayList<double[][]> possiblePartition = new ArrayList<double[][]>();
		ArrayList<Double> possibleCutPoints = new ArrayList<Double>();

		for (int i = first; i < (lastPlusOne - 1); i++) {
			counts[0][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
			counts[1][(int) instances.instance(i).classValue()] -= instances.instance(i).weight();
			if (instances.instance(i).value(attIndex) < instances.instance(i + 1).value(attIndex)) {
				currentCutPoint = (instances.instance(i).value(attIndex) + instances.instance(i + 1).value(attIndex))
						/ 2.0;
				possibleCutPoints.add(currentCutPoint);
				index.add(i);

				double[][] tempedCounts = new double[2][instances.numClasses()];
				for (int j = 0; j < tempedCounts.length; j++) {
					System.arraycopy(counts[j], 0, tempedCounts[j], 0, counts[j].length);
				}
				possiblePartition.add(tempedCounts);
				numCutPoints++;
			}
		}

		/*
		 * we sample for cut points by building a probability distribution of all the
		 * possible cut points. First, we calculate the difference between IG and the
		 * MDL score. Second, for the points that difference less than zero, these
		 * points will not be selected, so we make the probability to be zero. If all
		 * the point do not meet the MDL criterion and firstflag = true, we normalize IG
		 * into a probability distribution, select one cut point and return. This is to
		 * make sure that we have at least one cut point. otherwise, we sample for the
		 * distribution.
		 */
		double difference = 0;
		double[] distribution = new double[possibleCutPoints.size()];
		for (int i = 0; i < possibleCutPoints.size(); i++) {

			if (m_UseKononenko) {
				difference = KononenkosMDL(priorCounts, possiblePartition.get(i), numInstances, numCutPoints);
			} else {
				difference = FayyadAndIranisMDL(priorCounts, possiblePartition.get(i), numInstances, numCutPoints);
			}

			// if for some cut points with IG values < threshold, set IG= 0
			if (difference <= 0) {
				distribution[i] = 0;
			} else {
				distribution[i] = difference;
			}
		}

		// if no cut point meet MDL, then normalize IG and sample one cut point from it,
		// assuming that we have at least one point
		if (Utils.sum(distribution) == 0) {
			if (firstFlag) {
				double[] informationGain = new double[possibleCutPoints.size()];
				for (int i = 0; i < possibleCutPoints.size(); i++) {
					informationGain[i] = this.calculateInformationGain(priorCounts, possiblePartition.get(i));
				}

				SUtils.normalize(informationGain);

				if (Utils.sum(informationGain) <= 0) {
					return null;
				} else {
					double p = Math.random();
					int indexIG = SUtils.cumulativeProbability(informationGain, p);

					bestCutPoint = possibleCutPoints.get(indexIG);
					double[] oneCutPoint = new double[1];
					oneCutPoint[0] = bestCutPoint;
					return oneCutPoint;
				}
			} else {
				return null;
			}
		}

		// normalize IG to get a multinomial probability distribution
		SUtils.normalize(distribution);
		int selectedIndex;

		double p = Math.random();
		selectedIndex = SUtils.cumulativeProbability(distribution, p);
		bestCutPoint = possibleCutPoints.get(selectedIndex);

		if (selectedIndex != 0) {
			left = cutPointsForSubset(instances, attIndex, first, index.get(selectedIndex) + 1, false);
		}

		if (selectedIndex != index.size() - 1) {
			right = cutPointsForSubset(instances, attIndex, index.get(selectedIndex) + 1, lastPlusOne, false);

		}
		// Merge cutpoints and return them
		if ((left == null) && (right) == null) {
			cutPoints = new double[1];
			cutPoints[0] = bestCutPoint;
		} else if (right == null) {
			cutPoints = new double[left.length + 1];
			System.arraycopy(left, 0, cutPoints, 0, left.length);
			cutPoints[left.length] = bestCutPoint;
		} else if (left == null) {
			cutPoints = new double[1 + right.length];
			cutPoints[0] = bestCutPoint;
			System.arraycopy(right, 0, cutPoints, 1, right.length);
		} else {
			cutPoints = new double[left.length + right.length + 1];
			System.arraycopy(left, 0, cutPoints, 0, left.length);
			cutPoints[left.length] = bestCutPoint;
			System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
		}

		return cutPoints;
	}

	public Instance discretize(Instance row) {

		int index = 0;
	    double[] vals = new double[row.numAttributes()];
	    // Copy and convert the values
	    for (int i = 0; i < row.numAttributes(); i++) {
	      if (m_DiscretizeCols.isInRange(i)
	        && getInputFormat().attribute(i).isNumeric()) {
	        int j;
	        double currentVal = row.value(i);
	        if (m_CutPoints[i] == null) {
	          if (row.isMissing(i)) {
	            vals[index] = Utils.missingValue();
	          } else {
	            vals[index] = 0;
	          }
	          index++;
	        } else {
	          if (!m_MakeBinary) {
	            if (row.isMissing(i)) {
	              vals[index] = Utils.missingValue();
	            } else {
	              for (j = 0; j < m_CutPoints[i].length; j++) {
	                if (currentVal <= m_CutPoints[i][j]) {
	                  break;
	                }
	              }
	              vals[index] = j;
	            }
	            index++;
	          } else {
	            for (j = 0; j < m_CutPoints[i].length; j++) {
	              if (row.isMissing(i)) {
	                vals[index] = Utils.missingValue();
	              } else if (currentVal <= m_CutPoints[i][j]) {
	                vals[index] = 0;
	              } else {
	                vals[index] = 1;
	              }
	              index++;
	            }
	          }
	        }
	      } else {
	        vals[index] = row.value(i);
	        index++;
	      }
	    }

	    Instance inst = null;
	    if (row instanceof SparseInstance) {
	      inst = new SparseInstance(row.weight(), vals);
	    } else {
	      inst = new DenseInstance(row.weight(), vals);
	    }

//	    copyValues(inst, false, row.dataset(), outputFormatPeek());
	    
		return inst;
	}
}
