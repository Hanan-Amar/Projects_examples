package filesprocessing.Parsing;

public class ParseBadCommandFileException extends Exception {
    @Override
    public String getMessage() {
        return "ERROR: Bad Command File\n";
    }
}

class ParseBadSectionHeaderException extends ParseBadCommandFileException {

}

class ParseOrderSectionMissingException extends ParseBadCommandFileException {

}




