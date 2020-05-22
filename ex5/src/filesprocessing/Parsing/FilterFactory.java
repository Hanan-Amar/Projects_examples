package filesprocessing.Parsing;

import filesprocessing.Toolbox.Conversation;
import filesprocessing.Subsections.Filter;
import filesprocessing.Subsections.Filter.FilterTypes;
import java.io.File;
import java.util.function.Predicate;

/**
 * filter instances factory.
 */
public class FilterFactory {

    /**
     * generates a filter in a given type and arguments
     * @param type FilterType of filter
     * @param args arguments of Subsections.
     * @param not bool indicating if to negate.
     * @return Filter instance.
     */
    static Filter generateFilter(FilterTypes type, String[] args, boolean not) {
        Filter filter=null;
        switch (type){
            case between:
            case greater_than:
            case smaller_than: {
                double[] limits = Conversation.convertToSizeFilterArgs(args);
                filter = getSizeFilter(type, limits);
            } break;
            case prefix:
            case contains:
            case file:
            case suffix:{
                filter = getNameFilter(type,args[0]);
            } break;
            case hidden:
            case writable:
            case executable: {
                filter = getAttributeFilter(type, Conversation.convertToBinaryFilterArgs(args));
            } break;
            case all:{ filter = getAllFilter(); } break;
        }
        if (not) return new Filter.NegatedFilter(filter);
        return filter;
    }

    /**
     * returns a File-Size Filter.
     * @param type either smaller_than, bigger_than, between.
     * @param limits the doubles limits arguments of the filter
     * @return Filter instance.
     */
    private static Filter getSizeFilter(FilterTypes type, double[] limits) {
        switch (type){
            case between: {
                return new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> Conversation.toKB(file.length()) >= limits[0] &&
                            Conversation.toKB(file.length()) <= limits[1]; }
                };
            }

            case greater_than: {
                return new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> Conversation.toKB(file.length()) > limits[0]; }
                };
            }

            case smaller_than: {
                return new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> Conversation.toKB(file.length()) < limits[0]; }
                };
            }
        }
        return null;
    }

    /**
     * returns a File-Name Filter.
     * @param type either file, contains, prefix, suffix
     * @param arg the suite string argument.
     * @return Filter instance.
     */
    private static Filter getNameFilter(FilterTypes type, String arg) {
        Filter filter = null;
        switch (type) {
            case file: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.getName().equals(arg); }
                }; break;
            }
            case contains: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.getName().contains(arg); }
                }; break;
            }
            case prefix: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.getName().startsWith(arg); }
                }; break;
            }
            case suffix: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.getName().endsWith(arg); }
                }; break;
            }
        }
        return filter;
    }

    /**
     * returns a File-Attribute Filter.
     * @param type either writable, executable, hidden.
     * @param hasAtt bool indicating filter arg.
     * @return Filter instance.
     */
    private static Filter getAttributeFilter(FilterTypes type, boolean hasAtt) {
        Filter filter = null;
        switch (type) {
            case writable: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.canWrite() == hasAtt; }
                }; break;
            }
            case executable: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.canExecute() == hasAtt; }
                }; break;
            }
            case hidden: {
                filter = new Filter() {
                    @Override
                    public Predicate<File> pass() { return file -> file.isHidden() == hasAtt; }
                };break;
            }
        }
        return filter;
    }

    /**
     * return the All filter.
     * @return Filter instance
     */
    static Filter getAllFilter() {
        return new Filter() {
            @Override
            public Predicate<File> pass() {
                return file -> true;
            }
        };
    }
}
