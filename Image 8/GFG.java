
class GFG {
    public static int n = 8, m = 8;
    public static double pi = 3.142857;

    // Function to find discrete cosine transform and print it
    static strictfp void dctTransform(int matrix[][]) {
        int i, j, k, l;
        // dct will store the discrete cosine transform
        double[][] dct = new double[m][n];
        double ci, cj, dct1, sum =0;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                // ci and cj depends on frequency as well as

                // number of row and columns of specified matrix
                if (i == 0)
                    ci = 1 / Math.sqrt(m);
                else
                    ci = Math.sqrt(2) / Math.sqrt(m);
                if (j == 0)
                    cj = 1 / Math.sqrt(n);
                else
                    cj = Math.sqrt(2) / Math.sqrt(n);
                // sum will temporarily store the sum of
                // cosine signals sum = 0;
                for (k = 0; k < m; k++) {
                    for (l = 0; l < n; l++) {
                        dct1 = matrix[k][l] *
                                Math.cos((2 * k + 1) * i * pi / (2 * m)) *
                                Math.cos((2 * l + 1) * j * pi / (2 * n));
                        sum = sum + dct1;
                    }
                }
                dct[i][j] = ci * cj * sum;
            }
        }
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
                System.out.printf("%f\t", dct[i][j]);
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int matrix[][] = { { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 },
                { 255, 255, 255, 255, 255, 255, 255, 255 } };
        dctTransform(matrix);
    }
}