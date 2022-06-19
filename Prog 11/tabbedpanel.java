import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class tabbedpanel extends JPanel {
    String tabs[] = { "One", "Two", "Three", "Four" };
    public JTabbedPane tabbedPane = new JTabbedPane();

    public tabbedpanel() {
        setLayout(new BorderLayout());
        for (int i = 0; i < tabs.length; i++)
            tabbedPane.addTab(tabs[i], null, createPane(tabs[i]));
        tabbedPane.setSelectedIndex(0);
        add(tabbedPane, BorderLayout.CENTER);
    }

    JPanel createPane(String s) {
        JPanel p = new JPanel();
        p.add(new JLabel(s));
        return p;
    }

    public static void main(String args[]) {
        JFrame frame = new JFrame("Tabbed Pane Example");
        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        frame.getContentPane().add(new tabbedpanel(), BorderLayout.CENTER);
        frame.setSize(400, 125);
        frame.setVisible(true);
    }
}
