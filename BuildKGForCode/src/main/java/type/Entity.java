package type;

import lombok.Data;

@Data
public class Entity {
    public Entity(String name, String idName, int id, Property property){
        this.name = name;
        this.idName = idName;
        this.id = id;
        this.property = property;
    }
    String name;
    String idName;
    int id;
    public static enum Property{
        CLASS,
        VARIABLE,
        METHOD,
        PARAMETER,
        MEMBER
    }
    Property property;
}
