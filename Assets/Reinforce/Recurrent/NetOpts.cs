
namespace Recurrent {

  public class NetOpts {

    public int architectureInputSize { get; set; }
    public int[] architectureHiddenUnits { get; set; }
    public int architectureOutputSize { get; set; }

    public float trainingAlpha { get; set; }
    public float trainingLossClamp { get; set; }
    
    public float trainingLoss { get; set; }

    public float otherMu { get; set; }
    public float otherStd { get; set; }

  }
}
