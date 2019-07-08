package MDL_R;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import Method.SmoothingMethod;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class TenFoldCV_MDLR {

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
	
	protected static SmoothingMethod method = SmoothingMethod.HGS;

	public static void main(String[] args) throws Exception {
	
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
		int nD = data.numInstances();
		int nA = data.numAttributes();
		int nC = data.numClasses();
		System.out.print("\t" + nD + "\t" + nA + "\t" + nC);
		long seed = 3071980;

		Random random = new Random(seed);
		double rmse = 0;
		double loss = 0;
		double time = 0;
		for (int i = 0; i < 5; i++) {

			wdBayesOnlinePYP_MDLR learner = new wdBayesOnlinePYP_MDLR();
			learner.set_m_S(m_S);
			learner.setK(m_K);
//			learner.setMEstimation(M_estimation);
			learner.setGibbsIteration(m_IterGibbs);
			learner.setBackoff(m_Backoff);
			learner.setTying(m_Tying);
			learner.setPrint(m_MVerb);
			learner.setSmoothingMethod(method);

			Evaluation_MDLR eva = new Evaluation_MDLR(data);
			eva.setSize(m_EnsembleSize);
			eva.crossValidateModel(learner, data, 2, random);
			
			rmse += eva.getRMSE();
			loss += eva.getError();
			time += eva.getTrainTime();

			
//			System.out.print("\t"+Utils.doubleToString(eva.rootMeanSquaredError(), 6, 4));
//			System.out.print("\t"+Utils.doubleToString(eva.errorRate(), 6,4));	
//			System.out.print("\t"+eva.getTrainTime()+"\n");

		}
		rmse /= 5;
		loss /= 5;
		time /= 5;

		System.out.print("\t" + Utils.doubleToString(rmse, 6, 4));
		System.out.print("\t" + Utils.doubleToString(loss, 6, 4));
		System.out.print("\t" + time + "\n");
		}
	}

	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
		}
		
		string = Utils.getOption('m', options);
		if (string.length() != 0) {
			if(string.equalsIgnoreCase("HGS")){
				method = SmoothingMethod.HGS;
			}else if(string.equalsIgnoreCase("HDP")) {
				method = SmoothingMethod.HDP;
			}else if(string.equalsIgnoreCase("MESTIMATION")) {
				method = SmoothingMethod.M_estimation;
			}else if(string.equalsIgnoreCase("LAPLACE")) {
				method = SmoothingMethod.LAPLACE;
			}
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

		Utils.checkForRemainingOptions(options);
	}

}
