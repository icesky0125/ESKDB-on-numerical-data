import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CsvToArff {

    public static void main(String[] args) {
        try{
 
       File csvFile = new File(args[0]);
       File saveFile = new File(args[1]);
       
       System.out.println(csvFile.getName());
        
        // load the CSV file (input file)
        CSVLoader loader = new CSVLoader();
        loader.setSource(csvFile);
        String [] options = new String[1];
        options[0]="-H";
        loader.setOptions(options);
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        System.out.println(data.numInstances()+"\t"+data.numAttributes()+"\t"+data.numClasses());
        // save as an  ARFF (output file)
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(saveFile);
        saver.writeBatch();
        }
        catch(Exception e){
        }
    }
}