import type.Entity;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.TypeSolver;
import com.github.javaparser.resolution.declarations.ResolvedFieldDeclaration;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class ASTProjectGenerator {
    public static void main(String[] args) throws Exception {
        Map<String, Entity> classMap;
        int classId = 0;

        TypeSolver typeSolver = new CombinedTypeSolver(
                new ReflectionTypeSolver(),
                //new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/")),
                new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/threads/src/main/java")),
                new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/controller/src/main/java"))
        );

        // 配置 JavaParser 使用符号解析
        ParserConfiguration parserConfiguration = new ParserConfiguration()
                .setSymbolResolver(new JavaSymbolSolver(typeSolver));
        //JavaParser javaParser = new JavaParser(parserConfiguration);
        // 指定项目源码的根目录
        SourceRoot sourceRoot = new SourceRoot(Paths.get("/Users/zhaowei/Desktop/code/wildfly-core-main"));
        sourceRoot.setParserConfiguration(parserConfiguration);
        // 解析源代码
        List<ParseResult<CompilationUnit>> parseResults = sourceRoot.tryToParseParallelized();

        // 遍历解析结果
        for (ParseResult<CompilationUnit> result : parseResults) {
            result.getResult().ifPresent(cu -> {
                cu.accept(new ClassVisitor(),null);
            });
        }
    }
    // 访问类名的 Visitor
    private static class ClassVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, Void arg) {
            String className = n.getNameAsString();

            // 获取包名（如果存在）
            Optional<String> packageName = n.findCompilationUnit()
                    .flatMap(CompilationUnit::getPackageDeclaration)
                    .map(pd -> pd.getNameAsString());

            // 输出完整类名
            if (packageName.isPresent()) {
                System.out.println("完整类名: " + packageName.get() + "." + className);
            } else {
                System.out.println("类名: " + className);
            }
            // 遍历并打印类的所有属性及其类型
            for (FieldDeclaration field : n.getFields()) {
                // 获取字段的解析类型
                try {
                    ResolvedFieldDeclaration resolvedField = field.resolve();
                    ResolvedType resolvedType = resolvedField.getType();

                    String fieldName = field.getVariables().get(0).getNameAsString();
                    String fieldType = resolvedType.describe();

                    System.out.println(" 属性: " + fieldName + " - 类型: " + fieldType);
                } catch (Exception e) {
                    System.out.println(" 无法解析字段: " + field.getVariables().get(0).getNameAsString());
                    System.out.println(e);
                }
            }
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
}
