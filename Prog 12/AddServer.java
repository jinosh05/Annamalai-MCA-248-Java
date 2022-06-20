import java.rmi.*;

public class AddServer {
    public static void main(String[] args) {
        try {
            AddServerImpl server = new AddServerImpl();
            Naming.rebind("registerme", server);
            System.out.println("Server is running...");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}