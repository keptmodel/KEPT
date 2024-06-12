package visitor;

import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import type.Entity;

import java.util.Map;

public class VisitorBase extends VoidVisitorAdapter<Void>{
    int idCount=0;
    Map<String, Entity> baseMap;
    public Map<String,Entity> getBaseMap(){
        return baseMap;
    }
    protected void AddEntity(String keyName, String entityName,String idName, Entity.Property property){
        Entity entity = new Entity(entityName,idName,idCount++,property);
        if(baseMap.containsKey(keyName)){
            throw new IllegalArgumentException("Key '" + keyName + "' should not exist in the map.");
        }
        baseMap.put(keyName,entity);
    }
}
