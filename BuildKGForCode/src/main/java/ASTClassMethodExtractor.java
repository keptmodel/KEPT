import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.io.FileNotFoundException;

public class ASTClassMethodExtractor {
    public static void main(String[] args) {
        try {
            // 读取 Java 源文件
//            String fileName = "C:\\Users\\13240\\Desktop\\【缺陷定位】新研究\\01 数据收集\\RAW_REPO\\" +
//                    "quarkus\\core\\builder\\src\\main\\java\\io\\quarkus\\builder\\ProduceFlags.java";
            String fileName = args[0];
            File file = new File(fileName);

            // 解析 Java 文件为抽象语法树
            CompilationUnit cu = StaticJavaParser.parse(file);

            // 提取类名
            cu.accept(new ClassVisitor(), null);

            // 提取方法名
            cu.accept(new MethodVisitor(), null);

            // 提取变量名
            // cu.accept(new VariableVisitor(), null);

            // 提取 import 语句
            cu.accept(new ImportVisitor(), null);

            // 提取 implementation 关系
            cu.accept(new ImplementationVisitor(), null);

            // 提取 extend 关系
            cu.accept(new ExtendVisitor(), null);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    // 访问类名的 Visitor
    private static class ClassVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, Void arg) {
            System.out.println("Class name: " + n.getName());
            super.visit(n, arg);
        }
    }

    // 访问方法名的 Visitor
    private static class MethodVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(MethodDeclaration n, Void arg) {
            System.out.println("Method name: " + n.getName());
            n.getParameters().forEach(param -> {
                System.out.println(" 参数类型: " + param.getType());
                System.out.println(" 参数名称: " + param.getName());
            });
            super.visit(n, arg);
        }

    }

    // 访问变量名的 Visitor
    private static class VariableVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(VariableDeclarator n, Void arg) {
            System.out.println("Variable name: " + n.getName());
            super.visit(n, arg);
        }
    }

    // 提取 import 语句
    private static class ImportVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(ImportDeclaration n, Void arg) {
            System.out.println("Import statement: " + n.getName());
            super.visit(n, arg);
        }
    }

    // 提取 implementation 关系
    private static class ExtendVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, Void arg) {
//            System.out.println("Extended types: " + n.getExtendedTypes());
            if (n.getExtendedTypes().isNonEmpty()) {
                for (ClassOrInterfaceType extendedType : n.getExtendedTypes()) {
                    System.out.println("Extended types: " + extendedType.getNameAsString());
                }
            }
            super.visit(n, arg);
        }
    }

    // 提取 extend 关系
    private static class ImplementationVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, Void arg) {
//            System.out.println("Implemented types: " + n.getImplementedTypes());
            if (!n.getImplementedTypes().isEmpty()) {
                for (ClassOrInterfaceType implementedType : n.getImplementedTypes()) {
                    System.out.println("Implemented types: " + implementedType.getNameAsString());
                }
            }
            super.visit(n, arg);
        }
    }
    private static class InstanceOfVisitor extends  VoidVisitorAdapter<Void> {

    }
}