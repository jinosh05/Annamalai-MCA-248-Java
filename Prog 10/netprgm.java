import java.net.*;

class netprgm {
    public static void main(String arg[]) throws MalformedURLException, UnknownHostException {

        URL hp = new URL("http://www.osborne.com/download");
        System.out.println("Protocol : " + hp.getProtocol());
        System.out.println("Port : " + hp.getPort());
        System.out.println("Host : " + hp.getHost());
        System.out.println("File : " + hp.getFile());
        System.out.println("Ext : " + hp.toExternalForm());
        InetAddress someHost;
        byte bytes[];
        int fourBytes[] = new int[4];
        if (arg.length == 0) {
            someHost = InetAddress.getLocalHost();

        } else {
            someHost = InetAddress.getByName(arg[0]);
        }

        System.out.println("Host'" + someHost.getHostName() + "'has address: ");
        bytes = someHost.getAddress();
        for (int i = 0; i < fourBytes.length; i++) {
            fourBytes[i] = bytes[i] & 255;

        }
        System.out.println(fourBytes[0] + "." + fourBytes[1] + "." + fourBytes[2] + "." + fourBytes[3]);
    }

}