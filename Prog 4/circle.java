import java.awt.*;
import java.applet.*;;

public class circle extends Applet {
    public void paint(Graphics g) {
        for (int i = 0; i < 5; i++) {
            if (i % 2 == 0) {
                g.drawOval(120, i * 60, 50, 50);
            } else {
                g.fillOval(120, i * 60, 50, 50);
            }
        }
    }
}
