package visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;
import type.Entity;
import type.Relation;


import java.util.List;
import java.util.Optional;

public class ClassRelationVisitor extends VisitorBase{
    public ClassRelationVisitor(GraphManager graphManager){
        super();
        this.graphManager = graphManager;
        this.relationHelper = new RelationHelper(graphManager);
    }
    RelationHelper relationHelper;
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
            int classId = graphManager.findByIdName(packageName.get() + "." + className, Entity.Property.CLASS).getId();
            //处理属性关系
            for (FieldDeclaration field : n.getFields()) {
                // 获取字段的解析类型
                String fieldName = field.getVariables().get(0).getNameAsString();
                int fieldId = graphManager.findByIdName(packageName.get()+"." + className+"."+fieldName, Entity.Property.MEMBER).getId();
                relationHelper.addRelationId2Id(classId,fieldId, Relation.Property.HAS);
                //todo: 使用 resolve 获取具体的关系，而不是现在粗略的关系
                //获取instanceof 关系
                Type fieldType = field.getElementType();
                relationHelper.addRelationId2Name(fieldId,String.valueOf(fieldType), Entity.Property.CLASS,Relation.Property.INSTANCEOF);
            }
            //处理继承关系
            if (n.getExtendedTypes().isNonEmpty()) {
                for (ClassOrInterfaceType extendedType : n.getExtendedTypes()) {
                    String extendedTypeName =  extendedType.getNameAsString();
                    relationHelper.addRelationId2Name(classId,extendedTypeName, Entity.Property.CLASS,Relation.Property.INHERITANCE);
                }
            }
            //处理实现关系
            if(n.getImplementedTypes().isNonEmpty()){
                for (ClassOrInterfaceType implementedType : n.getImplementedTypes()) {
                    String implementedTypeName =  implementedType.getNameAsString();
                    relationHelper.addRelationId2Name(classId,implementedTypeName, Entity.Property.CLASS,Relation.Property.IMPLEMENT);
                }
            }
            //处理方法
            for(MethodDeclaration method:n.getMethods()){
                String methodName = method.getNameAsString();
                int methodId = graphManager.findByIdName(packageName.get()+"." + className+"."+methodName, Entity.Property.METHOD).getId();
                relationHelper.addRelationId2Id(classId,methodId,Relation.Property.HAS);
                String returnTypeName = String.valueOf(method.getType());
                relationHelper.addRelationId2Name(methodId,returnTypeName,Entity.Property.CLASS,Relation.Property.RETURN);

                List<MethodCallExpr> methodCalls = method.findAll(MethodCallExpr.class);
                for (MethodCallExpr call : methodCalls) {

                    String callName = String.valueOf(call.getName());
                    relationHelper.addRelationId2Name(methodId, callName,Entity.Property.METHOD,Relation.Property.CALL);
                }

                for(Parameter param : method.getParameters()){
                    String paramName = String.valueOf(param.getName());
                    int paramId = graphManager.findByIdName(packageName.get()+"." + className+"."+methodName+"."+paramName, Entity.Property.PARAMETER).getId();
                    relationHelper.addRelationId2Id(methodId,paramId, Relation.Property.HASPARAMETER);

                    String paramTypeName = String.valueOf(param.getType());
                    relationHelper.addRelationId2Name(paramId,paramTypeName, Entity.Property.CLASS, Relation.Property.INSTANCEOF);
                }
                method.findAll(VariableDeclarator.class).forEach(var -> {
                    String varName = String.valueOf(var.getName());
                    int varId = graphManager.findByIdName(packageName.get()+"." + className+"."+methodName+"."+varName, Entity.Property.VARIABLE).getId();
                    relationHelper.addRelationId2Id(methodId,varId, Relation.Property.HASVARIABLE);

                    String varTypeName = String.valueOf(var.getType());
                    relationHelper.addRelationId2Name(varId,varTypeName, Entity.Property.CLASS, Relation.Property.INSTANCEOF);
                });

            }
        } else {
            System.out.println("CANNOT FIND FULL NAME");
        }
    }
}
