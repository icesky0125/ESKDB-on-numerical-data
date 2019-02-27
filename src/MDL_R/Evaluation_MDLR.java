package MDL_R;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class Evaluation_MDLR extends weka.classifiers.evaluation.Evaluation {

	double trainTime = 0;
	String discretizeMethod;
	private int m_EnsembleSize;
	double m_errorRate;
	double m_RMSE;
	
	private static final long serialVersionUID = 1L;

	public Evaluation_MDLR(Instances data) throws Exception {
		super(data);
	}

	public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random)
			throws Exception {
		
		// Make a copy of the data we can reorder
		data = new Instances(data);
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Instances test = data.testCV(numFolds, i);
			train.setClassIndex(train.numAttributes() - 1);
			int nc = train.numClasses();

			// discretization added by He Zhang
			double start = System.currentTimeMillis();
			wdBayesOnlinePYP_MDLR[] classifiers = new wdBayesOnlinePYP_MDLR[this.m_EnsembleSize];
			MDLR[] discretizer = new MDLR[this.m_EnsembleSize];
			Instances[] allTests = new Instances[this.m_EnsembleSize];
			// train MDLR and classifier
			for (int k = 0; k < m_EnsembleSize; k++) {

				discretizer[k] = new MDLR();
				discretizer[k].setInputFormat(train);
				discretizer[k].setUseBetterEncoding(true);

				Instances currentTrain = Filter.useFilter(train, discretizer[k]);
				currentTrain.setClassIndex(currentTrain.numAttributes() - 1);
				allTests[k] = Filter.useFilter(test, discretizer[k]);

				classifiers[k] = (wdBayesOnlinePYP_MDLR) AbstractClassifier.makeCopy(classifier);
				classifiers[k].set_m_S("ESKDB_R");
				classifiers[k].buildClassifier(currentTrain);
			}

			trainTime += (System.currentTimeMillis() - start);

			// testing 
			double mse = 0, error = 0;
			for (int j = 0; j < test.numInstances(); j++) {

				int x_C = (int) test.get(j).classValue();// true class label
				
				// test using all the classifiers
				double[] probs = new double[nc];
				for (int k = 0; k < m_EnsembleSize; k++) {

					Instance currentInst = allTests[k].get(j);
					double[] res = classifiers[k].distributionForInstance(currentInst);
					for (int c = 0; c < nc; c++) {
						probs[c] += res[c];
					}
				}
				// ------------------------------------
				// averaged probability distribution
				// ------------------------------------
				for (int c = 0; c < nc; c++) {
					probs[c] /= this.m_EnsembleSize;
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

						mse += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					error += 1;
				}
			}

			this.m_RMSE += Math.sqrt(mse / test.numInstances());
			this.m_errorRate += (error / test.numInstances());
			classifiers = null;
			discretizer = null;
			allTests = null;
		}

		this.m_RMSE /= numFolds;
		this.m_errorRate /= numFolds;
		this.trainTime /= numFolds;	
	}

	public double getError() {
		return this.m_errorRate;
	}

	public double getRMSE() {
		return this.m_RMSE;
	}

	public double getTrainTime() {
		return trainTime;
	}

	public String getDisMethod() {
		return this.discretizeMethod;
	}

	public void setDisMethod(String s) {
		this.discretizeMethod = s;
	}

	public void setSize(int size) {
		m_EnsembleSize = size;
	}
}
