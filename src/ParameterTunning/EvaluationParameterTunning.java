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

public class EvaluationParameterTunning extends weka.classifiers.evaluation.Evaluation {

	public EvaluationParameterTunning(Instances data) throws Exception {
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

			double start = System.currentTimeMillis();
			copiedClassifier.buildClassifier(train);
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
		if (classificationOutput != null) {
			classificationOutput.printFooter();
		}
	}

	public void crossValidateModel(int bin, Classifier classifier, Instances data, int numFolds, Random random,
			Object... forPrinting) throws Exception {
		// TODO Auto-generated method stub
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
		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
			Instances test = data.testCV(numFolds, i);

			// discretization added by He Zhang
			weka.filters.unsupervised.attribute.Discretize dis = new weka.filters.unsupervised.attribute.Discretize();
			dis.setInputFormat(train);
			dis.setUseEqualFrequency(true);
			dis.setBins(bin);
			train = Filter.useFilter(train, dis);
			test = Filter.useFilter(test, dis);

			copiedClassifier.buildClassifier(train);
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
		m_NumFolds = numFolds;
		if (classificationOutput != null) {
			classificationOutput.printFooter();
		}
	}
}
