import java.rmi.*;

public interface AddServerIntf extends Remote {
    int add(int x, int y) throws RemoteException;
}