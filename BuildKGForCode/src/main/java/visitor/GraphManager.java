package visitor;

import com.opencsv.bean.ColumnPositionMappingStrategy;
import com.opencsv.bean.StatefulBeanToCsv;
import com.opencsv.bean.StatefulBeanToCsvBuilder;
import type.Entity;
import type.Relation;

import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class GraphManager {
    public GraphManager(){
        for(Entity.Property property:Entity.Property.values()){
            nameMap.put(property,new HashMap<>());
            idNameMap.put(property,new HashMap<>());
        }
    }
    Set<Relation> relationSet = new HashSet<>();
    Map<Entity.Property,Map<String, List<Entity>>> nameMap = new HashMap<>();
    Map<Entity.Property,Map<String, Entity>> idNameMap = new HashMap<>();
    ArrayList<Entity> entityArray = new ArrayList<>();
//    void addMethod(String name,String idName){
//        if(idNameMap.get(Entity.Property.METHOD).containsKey(idName)){
//            throw new IllegalArgumentException(idName+" idName Repeated!");
//        }
//        addEntity(name,idName,Entity.Property.METHOD);
//    }
//    void addMember(String name,String idName){
//        if(idNameMap.get(Entity.Property.MEMBER).containsKey(idName)){
//            return;
//        }
//        addEntity(name,idName, Entity.Property.MEMBER);
//    }
//    void addParameter(String name,String idName){
//        if(idNameMap.get(Entity.Property.PARAMETER).containsKey(idName)){
//            throw new IllegalArgumentException(idName+" idName Repeated!");
//        }
//        addEntity(name,idName, Entity.Property.PARAMETER);
//    }
//    void addVariable(String name,String idName){
//        if(idNameMap.get(Entity.Property.VARIABLE).containsKey(idName)){
//            throw new IllegalArgumentException(idName+" idName Repeated!");
//        }
//        addEntity(name,idName, Entity.Property.VARIABLE);
//    }
//    void addClass(String name,String idName){
//        if(idNameMap.get(Entity.Property.CLASS).containsKey(idName)){
//            return;
//        }
//        //if(nameMap.containsKey(name)){
//        //    throw new IllegalArgumentException(name+" Name Repeated!");
//        //}
//        addEntity(name,idName,Entity.Property.CLASS);
//    }
    public void addEntity(String name,String idName,Entity.Property property){
        if(idNameMap.get(property).containsKey(idName)){
            return;
        }
        Entity entity = new Entity(name,idName,entityArray.size(),property);
        entityArray.add(entity);
        if(nameMap.get(property).containsKey(name)){
            List<Entity> l = nameMap.get(property).get(name);
            l.add(entity);
        }
        else{
            List<Entity> l = new ArrayList<Entity>();
            l.add(entity);
            nameMap.get(property).put(name,l);
        }
        idNameMap.get(property).put(idName,entity);
    }

    //boolean hasName(String name){
    //    return nameMap.containsKey(name);
    //}
    //boolean hasIdName(String idName){
    //    return idNameMap.containsKey(idName);
    //}
    Entity findByIdName(String idName,Entity.Property property){
        return idNameMap.get(property).getOrDefault(idName, null);
    }
    List<Entity> findByName(String name,Entity.Property property){
        return nameMap.get(property).getOrDefault(name,null);
    }
    void addRelation(int head,int tail,Relation.Property property){
        Relation relation = new Relation(head,tail,property);
        relationSet.add(relation);
    }
    private static <T> void writeListToCsv(List<T> list, Class<T> clazz, String filePath) {
        // 设置 CSV 列的位置策略
        //ColumnPositionMappingStrategy<T> strategy = new ColumnPositionMappingStrategy<>();
        //strategy.setType(clazz);

        try (Writer writer = Files.newBufferedWriter(Paths.get(filePath))) {
            // 创建 StatefulBeanToCsv 对象
            StatefulBeanToCsv<T> beanToCsv = new StatefulBeanToCsvBuilder<T>(writer)
                    .build();

            // 将列表写入 CSV
            for (int i = 0; i < list.size(); i += 10000) {
                int end = Math.min(list.size(), i + 10000);
                List<T> batch = list.subList(i, end);
                beanToCsv.write(batch);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void saveCSV(String path){
        String entityPath = path+"/code_entity.csv";
        String relationPath = path+"/code_relation.csv";
        writeListToCsv(entityArray, Entity.class,entityPath);
        writeListToCsv(new ArrayList<Relation>(relationSet), Relation.class,relationPath);
    }
}
