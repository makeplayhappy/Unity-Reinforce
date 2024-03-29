namespace Recurrent{
    public class RandMat : Mat {
    /**
    * 
    * @param rows length of Matrix
    * @param cols depth of Matrix
    * @param mu Population mean for initialization
    * @param std Standard deviation for initialization
    */
    public RandMat(int rows, int cols, float mu, float std) : base(rows, cols) {
      
      Utils.fillRandn(ref w, mu, std);
    }
  }
}
