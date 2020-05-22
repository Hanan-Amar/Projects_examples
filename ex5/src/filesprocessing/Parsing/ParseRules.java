package filesprocessing.Parsing;

import filesprocessing.Toolbox.ValidInputChecks;
import filesprocessing.Subsections.Filter;
import filesprocessing.Subsections.Order;
import java.util.function.Predicate;

/**
 * represents the rules to parse a single subSection.
 * @param <T> enum types of subsection.
 */
public class ParseRules<T extends Enum> {

    /**
     * size filter validation checks.
     */
    private static final Predicate<String[]> sizeFiltersValids = doubles -> {
        for (String d: doubles) { if(!ValidInputChecks.isNonNegative(d)) return false; }
        if (doubles.length > 1) return ValidInputChecks.isRange(doubles[0], doubles[1]);
        return true;};

    /**
     * binary filter validation checks.
     */
    private static final Predicate<String[]> binaryFiltersValids = str -> ValidInputChecks.isBinaryValue(str[0]);
    /**
     * empty validation checks
     */
    private static final Predicate<String[]> emptyValids = str -> true;

    /**
     * the available Subsections parse rules
     */
    private static final ParseRules[] filtersParseRules = {
            new ParseRules<>(Filter.FilterTypes.greater_than, 1, sizeFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.smaller_than, 1, sizeFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.between, 2, sizeFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.executable, 1, binaryFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.writable, 1, binaryFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.hidden, 1, binaryFiltersValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.suffix, 1, emptyValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.prefix, 1, emptyValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.file, 1, emptyValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.contains, 1, emptyValids),
            new ParseRules<Filter.FilterTypes>(Filter.FilterTypes.all, 0, emptyValids)
    };

    /**
     * the available orders parse rules
     */
    private static final ParseRules[] ordersParseRules = {
            new ParseRules<>(Order.OrderTypes.abs, 0, emptyValids),
            new ParseRules<>(Order.OrderTypes.size, 0, emptyValids),
            new ParseRules<>(Order.OrderTypes.type, 0, emptyValids),
    };

    /**
     * matches a string typename to its suitable Filter rules.
     * @param typeName string name of type.
     * @return ParseRules instance if found, otherwise null.
     */
    static ParseRules getFilterRulesByType(String typeName){

        for (ParseRules p: filtersParseRules) {
            if (p.getType().name().equals(typeName)) {
                    return p;
            }
        }
        return null;
    }

    /**
     * matches a string typename to its suitable Order rules.
     * @param typeName string name of type.
     * @return ParseRules instance if found, otherwise null.
     */
    static ParseRules getOrderRulesByType(String typeName){
        for (ParseRules p: ordersParseRules) {
            if (p.getType().name().equals(typeName))
                return p;
        }
        return null;
    }


    /**
     * the number of expected arguments.
     */
    private int amountOfArgs;
    /**
     * the validation checks to run over the arguments.
     */
    private Predicate<String[]> validChecks;
    /**
     * the type of the subsection.
     */
    private T type;

    /**
     * ParseRules constructor (PRIVATE)
     * @param type the type of subsection
     * @param amountOfArgs number of arguments
     * @param validChecks predicate validation checks.
     */
    private ParseRules(T type, int amountOfArgs, Predicate<String[]> validChecks) {
        this.type = type;
        this.amountOfArgs = amountOfArgs;
            this.validChecks = validChecks;
    }

    /**
     * type getter.
     * @return returns the type of the subsection
     */
    T getType() {
        return type;
    }

    /**
     * evaluate the validation checks and number of arguments.
     * @param args args to evaluate.
     * @return true if all valid.
     */
    boolean doValidChecks(String[] args){
        return (args.length == amountOfArgs) && validChecks.test(args);
    }
}
