import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;

class Powerlaw {
    BufferedImage image;
    int width;
    int height;

    public Powerlaw() {
        try {
            File input = new File("nature.jpeg");
            image = ImageIO.read(input);
            width = image.getWidth();
            height = image.getHeight();
            double gamma;
            gamma = 1.25;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Color c = new Color(image.getRGB(j, i));
                    int red = (int) (c.getRed() * 0.299);
                    int green = (int) (c.getGreen() * 0.587);
                    int blue = (int) (c.getBlue() * 0.114);
                    int gray = (red + green + blue) / 3;
                    int gray1 = (int) Math.pow(gray, gamma);
                    // System.out.println("Gray"+gray1);
                    Color newColor = new Color(gray1, gray1, gray1);
                    image.setRGB(j, i, newColor.getRGB());
                }
            }
            File ouptut = new File("powerlaw1.jpg");
            ImageIO.write(image, "jpg", ouptut);
        } catch (Exception e) {
        }
    }

    static public void main(String args[]) throws Exception {
        Powerlaw obj = new Powerlaw();
    }
}