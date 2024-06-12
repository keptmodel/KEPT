import org.lambda3.graphene.core.Graphene;
import org.lambda3.graphene.core.relation_extraction.model.RelationExtractionContent;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NLPTest {
    public static void main(String[] args) {
        Graphene graphene = new Graphene();
        String filePath = "path_to_your_file.txt"; // 替换为你的文件路径
        List<String> lines = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        String text ="Adding the bootable JAR Maven plugin to your pom file.\n" +
                "This is done by adding an extra build step to your application deployment Maven pom.xml file.\n" +
                "The next chapter covers the plugin configuration items that are required to identify the WildFly server version and content.";
        RelationExtractionContent rec = graphene.doRelationExtraction(text, true, false);
        // ### OUTPUT AS RDFNL #####
// default
        String defaultRep = rec.defaultFormat(false); // set **true** for resolved format

// flat
        String flatRep = rec.flatFormat(false); // set **true** for resolved format

// ### OUTPUT AS PROPER RDF (N-Triples) ###
        String rdf = rec.rdfFormat();

// ### SERIALIZE & DESERIALIZE ###
        //RelationExtractionContent.serializeToJSON(new File("file.json"));
        //RelationExtractionContent loaded = RelationExtractionContent.deserializeFromJSON(new File("file.json"), RelationExtractionContent.class);

    }
}
