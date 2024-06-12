import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;

import java.io.File;
import java.io.FileNotFoundException;

public class ASTGenerator {
    public static void main(String[] args) {
        try {
            // 读取 Java 源文件
            File file = new File("YourJavaFile.java");

            // 解析 Java 文件为抽象语法树
            CompilationUnit cu = StaticJavaParser.parse(file);

            // 遍历抽象语法树中的所有节点
            cu.walk(node -> {
                // 获取节点类型并打印
                String nodeType = node.getClass().getSimpleName();
                System.out.println(nodeType + " : " + node.toString());
            });

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
