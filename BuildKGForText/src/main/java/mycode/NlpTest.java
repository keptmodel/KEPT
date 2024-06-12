package mycode;

import de.uni_mannheim.minie.MinIE;
import de.uni_mannheim.minie.annotation.AnnotatedProposition;
import de.uni_mannheim.utils.coreNLP.CoreNLPUtils;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NlpTest {
    public static void main(String[] args){
        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();

        // Input sentence
        //String sentence = "For a domain the active domain.xml and host.xml histories are kept in jboss.domain.config.dir /domain_xml_history and jboss.domain.config.dir /host_xml_history. ";
        // Generate the extractions (With SAFE mode)
        String filePath = "test.txt"; // 替换为你的文件路径
        List<String> lines = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
                MinIE minie = new MinIE(line, parser, MinIE.Mode.SAFE);
                System.out.println("\nInput sentence: " + line);
                System.out.println("=============================");
                for (AnnotatedProposition ap: minie.getPropositions()) {
                    System.out.println("\tTriple: " + ap.getTripleAsString());
                    System.out.print("\tFactuality: " + ap.getFactualityAsString());
                    if (ap.getAttribution().getAttributionPhrase() != null)
                        System.out.print("\tAttribution: " + ap.getAttribution().toStringCompact());
                    else
                        System.out.print("\tAttribution: NONE");
                    System.out.println("\n\t----------");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Print the extractions

    }
}
