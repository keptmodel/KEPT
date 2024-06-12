package type;

import lombok.Data;

@Data
public class Relation {
    public Relation(int head,int tail,Property property){
        this.head = head;
        this.tail = tail;
        this.property = property;
    }
    public enum Property{
        HAS,
        INSTANCEOF,
        INHERITANCE,
        RETURN,
        CALL,
        IMPLEMENT,
        HASPARAMETER,
        HASVARIABLE
    };
    Property property;
    int head;
    int tail;
}
