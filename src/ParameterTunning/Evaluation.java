package ParameterTunning;

import java.io.File;
import java.util.Random;

import org.apache.commons.math3.util.MathArrays.Position;

import ESKDB.wdBayesOnlinePYP;
import MDL_R.MDLR;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.attribute.*;

public class Evaluation extends weka.classifiers.evaluation.Evaluation {

	double trainTime = 0;
	String discretizeMethod;
	String m_MDLVersion;

	public Evaluation(Instances data) throws Exception {
		super(data);
		// TODO Auto-generated constructor stub
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random)
			throws Exception {
		crossValidateModel(classifier, data, numFolds, random, new Object[0]);
	}

	/**
	 * Performs a (stratified if class is nominal) cross-validation for a classifier
	 * on a set of instances. Performs a deep copy of the classifier before each
	 * call to buildClassifier() (just in case the classifier is not initialized
	 * properly).
	 *
	 * @param classifier  the classifier with any options set.
	 * @param data        the data on which the cross-validation is to be performed
	 * @param numFolds    the number of folds for the cross-validation
	 * @param random      random number generator for randomization
	 * @param forPrinting varargs parameter that, if supplied, is expected to hold a
	 *                    weka.classifiers.evaluation.output.prediction.AbstractOutput
	 *                    object or a StringBuffer for model output
	 * @throws Exception if a classifier could not be generated successfully or the
	 *                   class is not defined
	 */
	public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random,
			Object... forPrinting) throws Exception {

		// Make a copy of the data we can reorder
		data = new Instances(data);
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// We assume that the first element is a
		// weka.classifiers.evaluation.output.prediction.AbstractOutput object
		AbstractOutput classificationOutput = null;
		if (forPrinting.length > 0 && forPrinting[0] instanceof AbstractOutput) {
			// print the header first
			classificationOutput = (AbstractOutput) forPrinting[0];
			classificationOutput.setHeader(data);
			classificationOutput.printHeader();
		}
		System.out.print('\t');
		// Do the folds
		for (int i = 0; i < numFolds; i++) {

			System.out.print(i);
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
			Instances test = data.testCV(numFolds, i);

			// discretization added by He Zhang

			switch (discretizeMethod) {
			case "EqualFrequency":
				double bestRMSE = Double.MAX_VALUE;
				int[] bins = { 5, 10, 15,20};
				int bestBin = 0;

				for (int z = 0; z < bins.length; z++) {

					Instances currentData = new Instances(train);

					EvaluationParameterTunning eva = new EvaluationParameterTunning(currentData);
					eva.crossValidateModel(bins[z], classifier, currentData, 3, new Random(25011990));
					double currentRmse = eva.rootMeanSquaredError();
//					System.out.print("\t" + Utils.doubleToString(eva.rootMeanSquaredError(), 6, 4));
//					System.out.print("\t" + Utils.doubleToString(eva.errorRate(), 6, 4));

					if (bestRMSE > currentRmse) {
						bestRMSE = currentRmse;
						bestBin = bins[z];
					}
//					System.out.println();
				}
//				System.out.println("best bin is "+ bestBin);
				
				weka.filters.unsupervised.attribute.Discretize dis = new weka.filters.unsupervised.attribute.Discretize();
				dis.setInputFormat(train);
				dis.setUseEqualFrequency(true);
				dis.setBins(bestBin);
				train = Filter.useFilter(train, dis);
				test = Filter.useFilter(test, dis);

				break;
			case "MDL":
				weka.filters.supervised.attribute.Discretize disMDL = new weka.filters.supervised.attribute.Discretize();
				disMDL.setUseBetterEncoding(true);
				disMDL.setInputFormat(train);

				train = Filter.useFilter(train, disMDL);
				test = Filter.useFilter(test, disMDL);

				break;
			case "MDLOneHot":

				// first MDL
				disMDL = new weka.filters.supervised.attribute.Discretize();
				disMDL.setUseBetterEncoding(true);
				disMDL.setInputFormat(train);
				train = Filter.useFilter(train, disMDL);
				test = Filter.useFilter(test, disMDL);
				ArffSaver saver = new ArffSaver();
				saver.setFile(new File("mdl.arff"));
				saver.setInstances(train);
				saver.writeBatch();

				// second one hot encoding
				NominalToBinary oneHotEncoding = new NominalToBinary();
				oneHotEncoding.setInputFormat(train);
				train = Filter.useFilter(train, oneHotEncoding);
				test = Filter.useFilter(test, oneHotEncoding);
				saver = new ArffSaver();
				saver.setFile(new File("onehot.arff"));
				saver.setInstances(train);
				saver.writeBatch();

				// convert numeric to nominal {0,1}
				NumericToNominal convert = new NumericToNominal();
				convert.setInputFormat(train);
				train = Filter.useFilter(train, convert);
				test = Filter.useFilter(test, convert);
				saver = new ArffSaver();
				saver.setFile(new File("final.arff"));
				saver.setInstances(train);
				saver.writeBatch();

//	        boolean[] position = new boolean[train.numAttributes()-1];
//	        for(int z = 0; z < train.numAttributes()-1; z++) {
//	        	if(train.attribute(z).numValues() == 1 ) {
//	        		position[z] = true;
//	        	}
//	        }
//	        
//	        int count = 0;
//	        for(int z = 0; z < position.length; z++) {
//				 if(position[z]) {
//					 train.deleteAttributeAt(z-count);
//					 count++;
//				 }
//		     }
//	        
//	        count = 0;

//			 for(int z = 0; z < position.length; z++) {
//				 if(position[z]) {
//					 test.deleteAttributeAt(z-count);
//					 count++;
//				 }
//		     }

				break;
			case "MDLR":
				MDLR disMDLR = new MDLR();
				disMDLR.setInputFormat(train);
				disMDLR.setUseBetterEncoding(true);
				train = Filter.useFilter(train, disMDLR);
				test = Filter.useFilter(test, disMDLR);
				break;

			default:
				break;
			}

			double start = System.currentTimeMillis();
			copiedClassifier.buildClassifier(train);
			trainTime += System.currentTimeMillis() - start;
			if (classificationOutput == null && forPrinting.length > 0) {
				((StringBuffer) forPrinting[0])
						.append("\n=== Classifier model (training fold " + (i + 1) + ") ===\n\n" + copiedClassifier);
			}

			if (classificationOutput != null) {
				evaluateModel(copiedClassifier, test, forPrinting);
			} else {
				evaluateModel(copiedClassifier, test);
			}
		}
		System.out.print("\t");
		m_NumFolds = numFolds;
		trainTime /= m_NumFolds;
		if (classificationOutput != null) {
			classificationOutput.printFooter();
		}
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

	public void setMDLVersion(String m) {
		// TODO Auto-generated method stub
		this.m_MDLVersion = m;
	}
}
