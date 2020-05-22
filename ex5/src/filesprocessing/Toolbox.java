package filesprocessing;

import filesprocessing.Parsing.CommandFileParser;
import filesprocessing.Subsections.Order;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * help tools for the package.
 */
public class Toolbox {

    /**
     * parse arguments validation tools.
     */
    public static class ValidInputChecks {
        /**
         * checks if given string represents a non-negative double.
         *
         * @param s string to check
         * @return true if non negative.
         */
        public static boolean isNonNegative(String s) {
            return Double.parseDouble(s) >= 0;
        }

        /**
         * checks if given bounds represent a legal range.
         * @param low string represents low bound
         * @param high string represents high bound
         * @return true if legal.
         */
        public static boolean isRange(String low, String high){
            return Double.parseDouble(low) <= Double.parseDouble(high);
        }

        /**
         * checks if given value is "YES" or "NO"
         * @param s string to check
         * @return true if either one of the options.
         */
        public static boolean isBinaryValue(String s){
            return s.equals(CommandFileParser.FILTER_BINARY_TURE) || s.equals(CommandFileParser.FILTER_BINARY_FALSE);
        }
    }

    /**
     * conversation tools.
     */
    public static class Conversation {

        /**
         * converts string array to double array.
         * @param args string args.
         * @return double args.
         */
        public static double[] convertToSizeFilterArgs(String[] args){
            double[] lims = new double[args.length];
            for (int i=0; i < args.length; i++) lims[i] = Double.parseDouble(args[i]);
            return lims;
        }

        /**
         * converts string representing an binary array arg to boolean.
         * @param args string arg
         * @return boolean arg
         */
        public static boolean convertToBinaryFilterArgs(String[] args){
            return args[0].equals(CommandFileParser.FILTER_BINARY_TURE);
        }

        /**
         * strips type name and suffix (if exists) and leaves actual filter args.
         * @param args args to strip.
         * @param excludeSuffix is suffix exists.
         * @return actual args.
         */
        public static String[] getActualArgs(String[] args, boolean excludeSuffix){ //ASSUMES NOT ALL!
            int lowInd = 1;
            int lastInd = excludeSuffix ? args.length-2 : args.length-1;
            String[] actual = new String[lastInd - lowInd +1];
            for (int i = 0;i < lastInd - lowInd +1; i++){
                actual[i] = args[lowInd + i];
            }
            return actual;
        }

        /**
         * returns files extention from filename
         * @param name name of file (excluding path)
         * @return string extension
         */
        public static String getExtention(String name){
            if (name.length() == 0) return "";
            if (name.charAt(0) == '.') return "";
            int ind = name.lastIndexOf('.');
            if (ind == -1) return "";
            return name.substring(ind);
        }

        /**
         * converts bytes to kilo-bytes.
         * @param bytes to convert
         * @return bytes in kb.
         */
        public static double toKB(long bytes){
            return bytes / 1024.0;
        }
    }

    /**
     * files tools.
     */
    public static class FilesTools{
        /**
         * returns file content as string.
         * @param name name of file.
         * @return string content.
         * @throws IOException if file reading failed.
         */
        public static String readFileContent(String name) throws IOException {
            String content = "";
            StringBuilder builder = new StringBuilder();
            BufferedReader buffer = new BufferedReader(new FileReader(name));
            String line = buffer.readLine();
            while (line != null) {
                builder.append(line + "\n");
                line = buffer.readLine();
            }
            content = builder.toString();

            if (content.length() == 0) return content;

            return content.substring(0, content.length() - 1);
        }

        /**
         * returns all files in a given directory (excluding subdirectories)
         * @param directoryPath directory path.
         * @return list of files.
         */
        public static ArrayList<File> getFilesInPath(String directoryPath){
            ArrayList<File> files = new ArrayList<>();
            for(File f: new File(directoryPath).listFiles()){
                if (f.isFile()) files.add(f);
            }
            return files;
        }

        /**
         * heap sort help function.
         * stores the heap structure.
         * @param ord Order to order by.
         * @param files files to order.
         * @param length leangth of heap.
         * @param i current position.
         */
        private static void heapify(Order ord, File[] files, int length, int i) {
            int l = (2*i)+1;
            int r = (2*i)+2;
            int max_ind = i;

            if (l < length && ord.isBefore(files[l], files[max_ind]) > 0) max_ind = l;
            if (r < length && ord.isBefore(files[r], files[max_ind]) > 0) max_ind = r;

            if (max_ind != i) {
                File temp = files[i];
                files[i] = files[max_ind];
                files[max_ind] = temp;
                heapify(ord, files, length, max_ind);
            }
        }

        /**
         * sorts an array using heapSort accordiong to order logic.
         * @param ord Order to ord by.
         * @param files files to order.
         */
        public static void heapSort(Order ord, File[] files) {
            if (files.length == 0) return;

            // Building the heap
            int length = files.length;
            // we're going from the first non-leaf to the root
            for (int i = (length / 2)-1; i >= 0; i--)
                heapify(ord,files, length, i);

            for (int i = length-1; i >= 0; i--) {
                File temp = files[0];
                files[0] = files[i];
                files[i] = temp;

                heapify(ord,files, i, 0);
            }
        }
    }
}

