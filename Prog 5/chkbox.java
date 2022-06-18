import java.awt.*;
import java.awt.event.*;
import java.applet.*;

public class chkbox extends Applet implements ItemListener {
    Checkbox cb1, cb2, cb3;
    String msg = "";

    public void init() {

        // Labels
        cb1 = new Checkbox("Science");
        cb2 = new Checkbox("Management");
        cb3 = new Checkbox("Arts");

        // Adding to UI
        add(cb1);
        add(cb2);
        add(cb3);

        // Adding Item Listeners
        cb1.addItemListener(this);
        cb2.addItemListener(this);
        cb3.addItemListener(this);
    }

    public void itemStateChanged(
            ItemEvent ie) {
        repaint();
    }

    public void paint(Graphics g) {
        msg = "Current State:";
        g.drawString(msg, 6, 80);
        msg = "Science :" + cb1.getState();
        g.drawString(msg, 6, 100);
        msg = "Management :" + cb2.getState();
        g.drawString(msg, 6, 120);
        msg = "Arts :" + cb3.getState();
        g.drawString(msg, 6, 140);
    }
}
