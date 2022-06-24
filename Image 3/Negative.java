import java.io.*;

import javax.imageio.ImageIO;

import java.awt.*;

import java.awt.image.BufferedImage;

class Negative {

    BufferedImage image;

    int width;

    int height;

    public Negative()

    {
        try {

            File input = new File("nature.jpeg");
            image = ImageIO.read(input);

            width = image.getWidth();

            height = image.getHeight();

            for (int i = 0; i < height; i++) {

                for (int j = 0; j < width; j++) {

                    Color c = new Color(image.getRGB(j, i));

                    int red = (int) (c.getRed() * 0.299);

                    int green = (int) (c.getGreen() * 0.587);

                    int blue = (int) (c.getBlue() * 0.114);

                    int gray = (red + green + blue) / 3;

                    gray = (255 - gray - 1);

                    System.out.println("Gray" + gray);

                    Color newColor = new Color(gray, gray, gray);

                    image.setRGB(j, i, newColor.getRGB());

                    System.out.println("Break");

                }
            }
            File ouptut = new File("negativeimage.jpg");

            ImageIO.write(image, "jpg", ouptut);

        } catch (Exception e) {
        }
    }

    static public void main(String args[]) throws Exception {
        Negative obj = new Negative();

    }
}