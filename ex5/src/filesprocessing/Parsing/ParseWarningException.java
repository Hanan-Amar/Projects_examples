package filesprocessing.Parsing;

public class ParseWarningException extends Exception {
    @Override
    public String getMessage() {
        return "Warning in line ";
    }
}

class ParseTypeWarning extends ParseWarningException {

}

class ParseArgumentsWarning extends ParseWarningException {

}