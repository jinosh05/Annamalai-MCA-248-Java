import java.lang.*;

class Exception_handle {
    public static void main(String[] args) {
        int a = 10, b = 5, c = 5, x, y;

        try {
            x = a / (b - c);
        } catch (ArithmeticException e) {
            System.out.println("DIVISION BY ZERO");
        }
        y = a / (b + c);
        System.out.println("y=" + y);
    }
}
