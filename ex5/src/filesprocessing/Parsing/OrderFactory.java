package filesprocessing.Parsing;

import filesprocessing.Toolbox.Conversation;
import filesprocessing.Subsections.Order;
import java.io.File;

/**
 * A factory of Order instnaces.
 */
public class OrderFactory {

    public static Order generateOrder(Order.OrderTypes type, boolean reversed) {
        Order ord = null;
        switch (type) {
            case abs:
                ord = new Order() {
                    @Override
                    public int isBefore(File file1, File file2) {
                        return file1.getAbsolutePath().compareTo(file2.getAbsolutePath());
                    }
                };
                break;
            case size:
                ord = new Order() {
                    @Override
                    public int isBefore(File file1, File file2) {
                        if (file1.length() == file2.length())
                            return file1.getAbsolutePath().compareTo(file2.getAbsolutePath());
                        if (file1.length() > file2.length()) return 1;
                        else return -1;
                    }
                };
                break;
            case type:
                ord = new Order() {
                    @Override
                    public int isBefore(File file1, File file2) {
                        String ext1 = Conversation.getExtention(file1.getAbsolutePath());
                        String ext2 = Conversation.getExtention(file2.getAbsolutePath());
                        if (ext1.equals(ext2))
                            return file1.getAbsolutePath().compareTo(file2.getAbsolutePath());
                        return ext1.compareTo(ext2);
                    }
                };
                break;
        }

        if (reversed) return new Order.ReverseOrder(ord);
        else return ord;
    }

    public static Order getDefaultOrder(){
        return generateOrder(Order.OrderTypes.abs, false);
    }
}
