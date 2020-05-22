package filesprocessing.Subsections;

import java.io.File;

/**
 * represents an order
 */
public interface Order {

    /**
     * the method used to order.
     * @param file1 file1 to compare
     * @param file2 file2 to compare
     * @return 1 if file1 is before file2, -1 if opposite, 0 if equal.
     */
    int isBefore(File file1, File file2);

    /**
     * a "reverse" order decorator class
     */
    class ReverseOrder implements Order {

        /**
         * order to decorate.
         */
        Order toReverse;

        /**
         * consturctor
         * @param order order to decorate.
         */
        public ReverseOrder(Order order){
            toReverse = order;
        }

        /**
         * decorated filter
         * @param file1 file1 to compare
         * @param file2 file2 to compare
         * @return 1 if file1 is before file2, -1 if opposite, 0 if equal.
         */
        @Override
        public int isBefore(File file1, File file2) {
            return -1*toReverse.isBefore(file1,file2);
        }
    }

    /**
     * represents all order types (all names allowed)
     */
    enum OrderTypes {
        abs,
        type,
        size
    }
}
