package DisOnTrain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;

public class DiscretizeOnTrainForSatellite {
	
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static String dataTain;
	private static String dataTest;
	
	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));
		setOptions(args);
	
		File trainFile = new File(dataTain);
		
		String strData = trainFile.getName().substring(trainFile.getName().lastIndexOf("/") + 1,
				trainFile.getName().lastIndexOf("."));
		
		Instances train = readFile2Instances(trainFile);
		train.setClassIndex(train.numAttributes() - 1);
		System.out.println("Dataset : " + strData);
//		System.out.println("data size \t" + N);
		System.out.println("Attribute size \t" + train.numAttributes());
		System.out.println("class size \t" + train.numClasses());
		
		
		File testFile = new File(dataTest);
		Instances test = readFile2Instances(testFile);
		
		weka.filters.supervised.attribute.Discretize disMDL = new weka.filters.supervised.attribute.Discretize();
		disMDL.setUseBetterEncoding(true);
		disMDL.setInputFormat(train);

		train = Filter.useFilter(train, disMDL);
		test = Filter.useFilter(test, disMDL);
		
		saveInstances2File(train, "satellite_discretized_train");
		saveInstances2File(test, "satellite_discretized_test");
		
		
		System.out.println("finished discretization" );
	}

	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			dataTain = string;
		}

		string = Utils.getOption('T', options);
		if (string.length() != 0) {
			dataTest = string;
		}

		Utils.checkForRemainingOptions(options);
	}
	
	private static Instances readFile2Instances(File trainFile) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(trainFile), BUFFER_SIZE), 10000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances resultInstances = structure;

		Instance row;
		while ((row = reader.readInstance(structure)) != null) {
			resultInstances.add(row);
		}
		return resultInstances;
	}

	private static void saveInstances2File(Instances data, String name) throws IOException {
		Instances dataSet = data;
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);

		File res = new File(name + ".arff");
		res.deleteOnExit();
		saver.setFile(res);
		saver.writeBatch();
	}

}
