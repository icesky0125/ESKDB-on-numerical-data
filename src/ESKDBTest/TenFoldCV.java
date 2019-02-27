package ESKDBTest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;
import ESKDB.wdBayesOnlinePYP;
import ParameterTunning.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class TenFoldCV {

	private static String data = "";
	private static String m_S = "KDB"; // -S (NB,KDB,ESKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static int m_EnsembleSize = 5; // -E
	private static boolean m_Backoff = true;
	private static int m_Tying = 2; // -L
	private static String m_disMethod;
	private static String m_MDLVersion;

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println(Arrays.toString(args));
		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

		File[] folder = sourceFile.listFiles();
		Arrays.sort(folder);
		for (int d = 0; d < folder.length; d++) {

			sourceFile = folder[d];
		
			String name = sourceFile.getName().substring(0, sourceFile.getName().indexOf("."));
//			System.out.println(" java -jar target/EnsembleSKDB1-0.0.1-SNAPSHOT-jar-with-dependencies.jar -t data/"+sourceFile.getName()+" -S ESKDB -K 5 -I 1000 -L 2 -E 20 -D> hdpresult/ESKDB_HDP_MDL/"+name +".txt & ");
			System.out.print(name);
//			File sourceFile  = source;
			BufferedReader reader = new BufferedReader(new FileReader(sourceFile));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			int nD =data.numInstances();
			int nA = data.numAttributes();
			int nC = data.numClasses();

			long seed = 25011990;

			Random random = new Random(seed);

			wdBayesOnlinePYP learner = new wdBayesOnlinePYP();
			learner.set_m_S(m_S);
			learner.setK(m_K);
			learner.setMEstimation(M_estimation);
			learner.setGibbsIteration(m_IterGibbs);
			learner.setEnsembleSize(m_EnsembleSize);
			learner.setBackoff(m_Backoff);
			learner.setM_Tying(m_Tying);
			learner.setPrint(m_MVerb);

			Evaluation eva = new Evaluation(data);
			eva.setDisMethod(m_disMethod);
			if(m_disMethod.equalsIgnoreCase("MDLR")) {
				eva.setMDLVersion(m_MDLVersion);
			}
			eva.crossValidateModel(learner, data, 10, random);
			
			System.out.print("\t"+nD+"\t"+nA+"\t"+nC);
			System.out.print("\t"+Utils.doubleToString(eva.rootMeanSquaredError(), 6, 4));
			System.out.print("\t"+Utils.doubleToString(eva.errorRate(), 6,4));	
			System.out.print("\t"+eva.getTrainTime()+"\n");
		}
	}

	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
		}

		m_MVerb = Utils.getFlag('V', options);
		M_estimation = Utils.getFlag('M', options);

		m_Backoff = Utils.getFlag('B', options);

		string = Utils.getOption('S', options);
		if (string.length() != 0) {
			m_S = string;
		}

		string = Utils.getOption('K', options);
		if (string.length() != 0) {
			m_K = Integer.parseInt(string);
		}

		String ML = Utils.getOption('L', options);
		if (ML.length() != 0) {
			m_Tying = Integer.parseInt(ML);
		}

		string = Utils.getOption('E', options);
		if (string.length() != 0) {
			m_EnsembleSize = Integer.parseInt(string);
		}

		string = Utils.getOption('I', options);
		if (string.length() != 0) {
			m_IterGibbs = Integer.parseInt(string);
		}
		
		string = Utils.getOption('D', options);
		if (string.length() != 0) {
			if(string.equalsIgnoreCase("MDL")) {
				m_disMethod = "MDL";
			}else if(string.equalsIgnoreCase("EF")) {
				m_disMethod = "EqualFrequency";
			}else if(string.equalsIgnoreCase("OneHot")) {
				m_disMethod = "MDLOneHot";
			}else if(string.equalsIgnoreCase("MDLR")) {
				m_disMethod = "MDLR";
			}
		}
		
		string = Utils.getOption('H', options);
		if (string.length() != 0) {
			if(string.equals("1")) {
				m_MDLVersion = "V1";
			}else if(string.equals("2")) {
				m_MDLVersion = "V2";
			}else if(string.equals("3")) {
				m_MDLVersion = "V3";
			}
		}
		
		Utils.checkForRemainingOptions(options);
	}

}
