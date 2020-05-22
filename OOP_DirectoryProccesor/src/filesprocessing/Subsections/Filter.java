package filesprocessing.Subsections;

import java.io.File;
import java.util.function.Predicate;

/**
 * represents a filter
 */
public interface Filter {
    /**
     * the method used to filter.
     * @return predicate instance which its pred.test(File f) method returns true if to include.
     */
    Predicate<File> pass();

    /**
     * a "not" filter decorator class
     */
    class NegatedFilter implements Filter{
        /**
         * filter to decorate.
         */
        Filter toNegate;

        /**
         * constructor
         * @param filter filter to decorate.
         */
        public NegatedFilter(Filter filter){
            toNegate = filter;
        }

        /**
         * decorated pass method.
         * @return predicate instance which its pred.test(File f) method returns true if to include.
         */
        @Override
        public Predicate<File> pass() {
            return toNegate.pass().negate();

        }
    }

    /**
     * represents all filter types (all names allowed)
     */
    enum FilterTypes {
        greater_than,
        between,
        smaller_than,
        file,
        contains,
        prefix,
        suffix,
        writable,
        executable,
        hidden,
        all;
    }
}
