package filesprocessing;

import filesprocessing.Subsections.Filter;
import filesprocessing.Subsections.Order;

import java.util.ArrayList;

/**
 * represents a section in a command file.
 */
public class Section {
    /**
     * the sections filter
     */
    private Filter filter;
    /**
     * the sections order
     */
    private Order order;
    /**
     * the section warnings produced from parsing.
     */
    private ArrayList<String> warnings;

    /**
     * default constructor
     */
    public Section() {
        warnings = new ArrayList<>();
    }

    /**
     * adds a warning to the section.
     *
     * @param warning warning msg to add.
     */
    public void addWarning(String warning) {
        warnings.add(warning);
    }

    /**
     * sets the sections filter.
     *
     * @param filter filter to set
     */
    public void setFilter(Filter filter) {
        this.filter = filter;
    }

    /**
     * sets the sections order.
     *
     * @param order order to set
     */
    public void setOrder(Order order) {
        this.order = order;
    }

    /**
     * filter getter
     *
     * @return filter instance
     */
    public Filter getFilter() {
        return filter;
    }

    /**
     * order getter
     *
     * @return order instance
     */
    public Order getOrder() {
        return order;
    }

    /**
     * prints all warnings as errors.
     */
    public void printWarnings() {
        for (String warn : warnings) System.err.println(warn);
    }


}
