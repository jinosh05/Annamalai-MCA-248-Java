
class pyramid {

    public static void main(String[] args) {
        int rows = 9; // Number of Rows we want to print

        // Printing the pattern

        for (int i = 1; i <= rows; i++) {// outer forloop
            for (int j = 1; j <= (rows - i) * 2; j++) {
                System.out.print(" ");// create initial space for pyramid shape
            }
            for (int k = i; k >= 1; k--) {// inner for loops
                System.out.print(" " + k);// create left half
            }
            for (int l = 2; l <= i; l++) {
                System.out.print(" " + l); // create right half
            } // end outer for loop
            System.out.println();
        }

    }
}
