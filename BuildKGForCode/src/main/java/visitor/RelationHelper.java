package visitor;

import type.Entity;
import type.Relation;

import java.util.List;

public class RelationHelper {
    GraphManager graphManager;
    public RelationHelper(GraphManager graphManager){
        this.graphManager = graphManager;
    }
    void addRelationId2Id(int head, int tail, Relation.Property property){
        graphManager.addRelation(head,tail,property);
    }
    void addRelationId2Name(int head,String name,Entity.Property tailProperty,Relation.Property property){
        Entity findedEntity = graphManager.findByIdName(name, tailProperty);
        if(findedEntity!=null){
            graphManager.addRelation(head,findedEntity.getId(),property);
            return;
        }

        List<Entity> list = graphManager.findByName(name,tailProperty);
        if(list==null) {
            graphManager.addEntity(name,name,tailProperty);
            list = graphManager.findByName(name,tailProperty);
        }
        if (list.size()>1){
            return ;
        }
        for (Entity entity:list) {
            graphManager.addRelation(head,entity.getId(),property);
        }
    }
    void addRelationId2StrId(int head,String name,String idName,Entity.Property tailProperty,Relation.Property property){
        Entity entity = graphManager.findByIdName(idName,tailProperty);
        if(entity==null){
            graphManager.addEntity(name,idName,tailProperty);
            entity = graphManager.findByIdName(idName,tailProperty);
        }
        graphManager.addRelation(head,entity.getId(),property);
    }


}
