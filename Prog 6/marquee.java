import java.awt.*;
import java.lang.*;
import java.applet.*;

public class marquee extends Applet implements Runnable {
    Thread t;
    int x, y, f;
    String msg;

    public void init() {
        x = 10;
        y = 100;
        f = 5;
        msg = "This is growing marquee";
    }

    public void start() {
        t = new Thread();
        t.start();
    }

    public void run() {
        for (;;) {
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                repaint();
            }
        }
    }

    public void stop() {
        t.stop();
    }

    public void paint(Graphics g) {
        Font newFont = new Font(
                "TimesRoman", Font.BOLD, f);
        g.setFont(newFont);
        FontMetrics fontMetrics = g.getFontMetrics();
        int width = fontMetrics.stringWidth(msg);
        g.drawString(msg, x, y);
        f += 2;
        if (f > 50) {
            f = 2;
        }
    }
}
