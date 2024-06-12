import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.resolution.TypeSolver;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;
import type.Entity;
import visitor.ClassRelationVisitor;
import visitor.ClassVisitor;
import visitor.GraphManager;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ASTProjectGenerator1 {
    public static List<Path> findDirectoriesByName(Path rootPath, String targetDirectoryName) throws IOException, IOException {
        List<Path> result = new ArrayList<>();
        Files.walkFileTree(rootPath, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                // Check if the current directory matches the targetDirectoryName
                if (dir.endsWith(targetDirectoryName)) {
                    result.add(dir);
                }
                return FileVisitResult.CONTINUE;
            }
        });
        return result;
    }
    public static void main(String[] args) throws Exception {
        Map<String, Entity> classMap;
        GraphManager graphManager = new GraphManager();
        int classId = 0;

        TypeSolver typeSolver = new CombinedTypeSolver(
                new ReflectionTypeSolver()
                //new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/")),
                //new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/threads/src/main/java")),
                //new JavaParserTypeSolver(new File("/Users/zhaowei/Desktop/code/wildfly-core-main/controller/src/main/java"))
        );

        // 配置 JavaParser 使用符号解析
        ParserConfiguration parserConfiguration = new ParserConfiguration()
                .setSymbolResolver(new JavaSymbolSolver(typeSolver));
        //JavaParser javaParser = new JavaParser(parserConfiguration);
        // 指定项目源码的根目录
        Path rootPath = Paths.get("/Users/zhaowei/code/logging-log4j2");
        String targetDirectoryName = "src";
        //String targetSrc = "src/test/java";
        //String targetTest = "src/test";
        List<Path> pathList = findDirectoriesByName(rootPath, targetDirectoryName);
        //List<Path> pathList1 = findDirectoriesByName(rootPath,targetSrc);
        //List<Path> pathList2 =  findDirectoriesByName(rootPath,targetTest);
        //pathList.forEach(System.out::println);
        //pathList.addAll(pathList1);
        //pathList.addAll(pathList2);
        List<SourceRoot> sourceRoots = new ArrayList<>();
        // 为每个源码根目录创建 SourceRoot 实例，并添加到列表
        for (Path path : pathList) {
            SourceRoot sourceRoot = new SourceRoot(path);
            sourceRoot.setParserConfiguration(parserConfiguration);
            sourceRoots.add(sourceRoot);
            System.out.println(sourceRoot);
        }
        ProgressBar pb1 = new ProgressBar("Entity Found", sourceRoots.size());
        ProgressBar pb2 = new ProgressBar("Relation Found", sourceRoots.size());
        // 遍历 SourceRoot 列表，解析所有目录
        for (SourceRoot sourceRoot : sourceRoots) {

            List<ParseResult<CompilationUnit>> parseResults = sourceRoot.tryToParse();
            for (ParseResult<CompilationUnit> result : parseResults) {
                result.getResult().ifPresent(cu -> {
                    cu.accept(new ClassVisitor(graphManager), null);
                });
            }
            pb1.step();
        }
        for (SourceRoot sourceRoot : sourceRoots) {
            List<ParseResult<CompilationUnit>> parseResults = sourceRoot.tryToParse();
            for (ParseResult<CompilationUnit> result : parseResults) {
                result.getResult().ifPresent(cu -> {
                    cu.accept(new ClassRelationVisitor(graphManager),null);
                });
            }
            pb2.step();
        }

        System.out.println("Save Graph to CSV==========");
        graphManager.saveCSV(".");
    }
    // 访问类名的 Visitor
}