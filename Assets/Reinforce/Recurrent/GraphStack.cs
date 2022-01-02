namespace Recurrent{

    //this stores the operations on the Backpropagation Stack, using a class os it's pass by reference
  public class GraphStack{
    public GraphOperations operation;
    public Mat m1;
    public int rowIndex;
    public Mat m2;
    public Mat matOut;

    public GraphStack(GraphOperations op, ref Mat m1_in, int indx, ref Mat m_out){
        operation = op;
        m1 = m1_in;
        rowIndex = indx;
        matOut = m_out;
    }

    public GraphStack(GraphOperations op, Mat m1_in, Mat m_out){
        operation = op;
        m1 = m1_in;
        matOut = m_out;
    }

    public GraphStack(GraphOperations op, Mat m1_in, Mat m2_in, Mat m_out){
        operation = op;
        m1 = m1_in;
        m2 = m2_in;
        matOut = m_out;
    }

  }
}