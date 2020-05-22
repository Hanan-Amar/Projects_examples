package filesprocessing;
import filesprocessing.Parsing.CommandFileParser;
import filesprocessing.Parsing.ParseBadCommandFileException;
import filesprocessing.Subsections.Filter;
import filesprocessing.Subsections.Order;
import filesprocessing.Toolbox.FilesTools;
import java.io.*;
import java.util.ArrayList;

/**
 * Filter and Order given directory according to given command file
 * command_file.flt is composed of many sections where each form is:
 *
 * FILTER                   (HEADER)
 * NAME#ARGS...#ARGS#NOT    (NOT is optional)
 * ORDER                    (HEADER)
 * NAME#REVERSED            (REVERSED is optional)
 *
 * results (filenames) for each section are printed.
 */
public class DirectoryProcessor {

    /**
     * invalid arguments error message
     */
    private static String ERR_INVALID_ARGS = "ERROR: Invalid arguments.\n";
    /**
     * reading command file error message
     */
    private static String ERR_READING_FILE = "ERROR: Reading file Failed\n";

    /**
     * filter and orders a given directory according to command file.
     * prints the result to stdout
     * @param args if form of [directory,commandfile]
     */
    public static void main(String[] args) {
        if (args.length != 2){ System.err.println(ERR_INVALID_ARGS); return;}
        String sourceDir = args[0];
        String commandFile = args[1];

        ArrayList<File> files = FilesTools.getFilesInPath(sourceDir);
        ArrayList<Section> sections;
        try{ sections = CommandFileParser.ParseCommandFile(FilesTools.readFileContent(commandFile)); }
        catch (IOException e) { System.err.println(ERR_READING_FILE); return; }
        catch (ParseBadCommandFileException e) { System.err.println(e.getMessage()); return; }

        for(Section s : sections) ExecuteSection(s, files);
    }

    /**
     * executes one section: Subsections, order, prints.
     * @param section section to execute.
     * @param files files to execute on.
     */
    private static void ExecuteSection(Section section, ArrayList<File> files) {
        File[] result = filter(section.getFilter(), files);
        order(section.getOrder(), result);
        section.printWarnings();
        for (File f:result) System.out.println(f.getName());
    }

    /**
     * Subsections files according to given filter.
     * @param filter filter to filter by.
     * @param files files to filter.
     * @return all files passed filter.
     */
    private static File[] filter(Filter filter, ArrayList<File> files){
        ArrayList<File> result = new ArrayList<>();
        for(File file: files) {
            if(filter.pass().test(file))
                result.add(file);
        }
        File[] array_result = new File[result.size()];
        for(int i=0; i < array_result.length; i++)
            array_result[i] = result.get(i);
        return array_result;

    }

    /**
     * orders files according to given order.
     * @param order order to order by.
     * @param files files to order.
     */
    private static void order(Order order, File[] files){
        FilesTools.heapSort(order, files);
    }
}
