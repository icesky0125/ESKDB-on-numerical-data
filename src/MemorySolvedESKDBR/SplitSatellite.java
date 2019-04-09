package MemorySolvedESKDBR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.commons.math3.random.MersenneTwister;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;

public class SplitSatellite {
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	
	public static void main(String[] args) throws Exception {
		
		String string = Utils.getOption('t', args);
		String data = "";
		if (string.length() != 0) {
			data = string;
		}
		
		File sourceFile = new File(data);
		
		if(!sourceFile.exists()) {
			System.out.println("not exist");
		}
		
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int N = getNumData(sourceFile, structure);

		String strData = sourceFile.getName().substring(sourceFile.getName().lastIndexOf("/") + 1,
				sourceFile.getName().lastIndexOf("."));
		System.out.println("Dataset : " + strData);
		System.out.println("data size \t" + N);
		System.out.println("Attribute size \t" + structure.numAttributes());
		System.out.println("class size \t" + nc);
		System.out.print(strData + "\t");
		
		createTrainTmpFile(sourceFile, structure,N);
		
	}

	
	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			nLines++;
		}
		return nLines;
	}

	public static void createTrainTmpFile(File sourceFile, Instances structure,int N) throws IOException {
		File outTrain = new File("satellite_train" + ".arff");
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(outTrain);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);
		
		File outTest = new File("satellite_test" + ".arff");
		ArffSaver fileSaverTest = new ArffSaver();
		fileSaverTest.setFile(outTest);
		fileSaverTest.setRetrieval(Saver.INCREMENTAL);
		fileSaverTest.setStructure(structure);
		
		int testSize = (N % 3 == 0) ? N / 3 : N / 3 + 1;
		int trainSize = N-testSize;
		if((testSize + trainSize) != N) {
			System.out.println("not equal");
		}

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);

		Instance current;
		int lineNo = 0;
		int test = 0;
		int train = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (lineNo < trainSize) {
				fileSaver.writeIncremental(current);
				train++;
			}else {
				fileSaverTest.writeIncremental(current);
				test++;
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		fileSaverTest.writeIncremental(null);
		System.out.println(train+"\t"+test);
		
	}
	
}
