import java.util.Scanner;

public class Threenum {
    public static void main(String args[]) {
        int a, b, c;
        System.out.println("Enter A,B & C integers: ");
        Scanner s = new Scanner(System.in);
        a = s.nextInt();
        b = s.nextInt();
        c = s.nextInt();
        if (a > b && a > c)
            System.out.println("A is BIG: " + a);
        else if (b > a && b > c)
            System.out.println("B is BIG: " + b);
        else if (c > a && c > b)
            System.out.println("C is BIG: " + c);
        else
            System.out.println("The numbers are same.");
    }
}