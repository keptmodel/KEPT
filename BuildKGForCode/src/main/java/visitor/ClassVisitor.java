package visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import type.Entity;

import java.util.Optional;

public class ClassVisitor extends VisitorBase{
    public ClassVisitor(GraphManager manager){
        super();
        this.graphManager = manager;
    }
    GraphManager graphManager;
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        super.visit(n, arg);
        String className = n.getNameAsString();

        // 获取包名（如果存在）
        Optional<String> packageName = n.findCompilationUnit()
                .flatMap(CompilationUnit::getPackageDeclaration)
                .map(pd -> pd.getNameAsString());

        // 输出完整类名
        if (packageName.isPresent()) {
            //System.out.println("完整类名: " + packageName.get() + "." + className);
            graphManager.addEntity(className,packageName.get()+"."+className, Entity.Property.CLASS);
            for (FieldDeclaration field : n.getFields()) {
                // 获取字段的解析类型
                try {

                    String fieldName = field.getVariables().get(0).getNameAsString();

                    //System.out.println(" 属性: " + fieldName );
                    graphManager.addEntity(fieldName,packageName.get()+"." + className+"."+fieldName, Entity.Property.MEMBER);
                } catch (Exception e) {
                    //System.out.println(" 无法解析字段: " + field.getVariables().get(0).getNameAsString());
                    //System.out.println(e);
                }
            }
            for(MethodDeclaration method:n.getMethods()){
                String methodName = method.getNameAsString();
                graphManager.addEntity(methodName,packageName.get()+"." + className+"."+methodName, Entity.Property.METHOD);
                //System.out.println("method: "+methodName);
                for(Parameter param : method.getParameters()){
                    String paramName = String.valueOf(param.getName());
                    graphManager.addEntity(paramName,packageName.get()+"." + className+"."+methodName+"."+paramName, Entity.Property.PARAMETER);
                    //System.out.println("Param: "+paramName);
                }
                method.findAll(VariableDeclarator.class).forEach(var -> {
                    String varName = String.valueOf(var.getName());
                    graphManager.addEntity(varName,packageName.get()+"." + className+"."+methodName+"."+varName, Entity.Property.VARIABLE);
                });
            }

        } else {
            //System.out.println("CANNOT FIND FULL NAME");
        }
    }
}
