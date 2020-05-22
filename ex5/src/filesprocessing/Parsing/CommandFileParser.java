package filesprocessing.Parsing;

import filesprocessing.Section;
import filesprocessing.Toolbox.Conversation;
import filesprocessing.Subsections.Filter;
import filesprocessing.Subsections.Order;
import java.util.ArrayList;

public class CommandFileParser {
    /**
     * the decoding of the sections.
     */
    private static String FILTER_SECTION_HEADER = "FILTER";
    private static String ORDER_SECTION_HEADER = "ORDER";
    private static String ORDER_REVERSE_SUFFIX = "REVERSE";
    private static String FILTER_NOT_SUFFIX = "NOT";
    public static String FILTER_BINARY_TURE = "YES";
    public static String FILTER_BINARY_FALSE = "NO";
    private static String ARGS_SEPARATOR = "#";
    private static String LINES_SEPARATOR = "\n";

    /**
     * parse command line and returns Filter instance.
     * @param command string in form of NAME#ARGS..#ARGS#NOT_SFFIX(optional)
     * @return Filter instance
     * @throws ParseWarningException if parsing failed due to bad name, bad args value/count.
     */
    private static Filter parseFilter(String command) throws ParseWarningException {
        String[] args = command.split(ARGS_SEPARATOR);

        if (args.length == 0) throw new ParseArgumentsWarning();                 //empty row, return ALL FILTER
        ParseRules rules = ParseRules.getFilterRulesByType(args[0]);
        if (rules == null) throw new ParseTypeWarning();                         //BAD FILTER NAME
        boolean not = (args[args.length-1].equals(FILTER_NOT_SUFFIX));           // CHECK IF TO NEGATE
        String[] actualArgs = Conversation.getActualArgs(args, not);             //strips args
        if (!rules.doValidChecks(actualArgs)) throw new ParseArgumentsWarning(); //value and amount args check
        return FilterFactory.generateFilter((Filter.FilterTypes)rules.getType(),actualArgs,not); //casting cannot fail, getFilterRulesByType method returns only right types.
    }

    /**
     * parse command line and returns Order instance.
     * @param command string in form of NAME#REVERSE_SFFIX(optional)
     * @return Order instance
     * @throws ParseWarningException if parsing failed due to bad name, bad args value/count.
     */
    private static Order parseOrder(String command) throws ParseWarningException{
        String[] args = command.split(ARGS_SEPARATOR);
        if (args.length == 0 || args.length > 2) throw new ParseWarningException();    //EMPTY LINE
        ParseRules rules = ParseRules.getOrderRulesByType(args[0]);
        if (rules == null) throw new ParseTypeWarning();                               //BAD ORDER NAME
        boolean reversed = (args[args.length-1].equals(ORDER_REVERSE_SUFFIX));         // CHECK IF TO REVERSE
        if (!reversed && args.length == 2) throw new ParseArgumentsWarning();          //bad suffix
        return OrderFactory.generateOrder((Order.OrderTypes)rules.getType(),reversed); //casting cannot fail, getOrderRulesByType method returns only right types.
    }

    /**
     * parses a command file into arrayList of Section, where each section is
     * built from Filter and Order.
     * @param content the string content if the commandfile
     * @return ArrayList of sections.
     * @throws ParseBadCommandFileException if parsing failed dut to bad file structure.
     */
    public static ArrayList<Section> ParseCommandFile(String content) throws ParseBadCommandFileException {
        String[] lines = content.split(LINES_SEPARATOR);
        if (lines.length == 1 && lines[0].equals("")) return new ArrayList<>();

        ArrayList<Section> sections = new ArrayList<>();

        for(int i = 0; i < lines.length; i+=4) {
            Section sec = new Section();

            //CHECK HEADER - LINE i
            if(!lines[i].equals(FILTER_SECTION_HEADER))
                throw new ParseBadSectionHeaderException(); //BAD SUB-SECTION HEADER

            //CREATE FILTER INSTANCE - LINE i + 1
            try { sec.setFilter( parseFilter(lines[i+1])); } //if i+1 dosenot exists -> bad file
            catch (IndexOutOfBoundsException e) { throw new ParseOrderSectionMissingException(); } //ORDER SUBSECTION DOSE NOT EXISTS
            catch (ParseWarningException e) {
                sec.setFilter(FilterFactory.getAllFilter());
                sec.addWarning(e.getMessage() + (i + 1 + 1));
            }

            //CHECK HEADER - LINE i+2
            if(i+2 >= lines.length || !lines[i+2].equals(ORDER_SECTION_HEADER))
                throw new ParseBadSectionHeaderException(); //BAD SUB-SECTION HEADER / DOSENT EXISTS

            //CREATE ORDER INSTANCE - LINE i + 3
            if (i+3 >= lines.length) { //no order args -> order: abs, break.
                sec.setOrder(OrderFactory.getDefaultOrder());
                sections.add(sec);
                break;
            } else if (lines[i+3].equals(FILTER_SECTION_HEADER)) {
                i--;
                sec.setOrder(OrderFactory.getDefaultOrder());
                sections.add(sec);
                continue;} //next line is new section


            try { sec.setOrder(parseOrder(lines[i+3])); } //ARGS EXISTS: parse args
            catch (ParseWarningException e) {
                sec.setOrder(OrderFactory.getDefaultOrder());
                sec.addWarning(e.getMessage() + (i + 3 + 1));
            }

            sections.add(sec);
        }
        return sections;
    }
}
