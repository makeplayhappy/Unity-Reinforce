//import { Mat, RandMat, Graph, NetOpts } from '.';


namespace Recurrent{
  public class Net : NetOpts {
    public Mat W1;
    public Mat b1;
    public Mat W2t;
    public Mat b2;

    /**
    * Generates a Neural Net instance from a pre-trained Neural Net JSON.
    * @param {{W1, b1, W2, b2}} opt Specs of the Neural Net.
    */
    Net(Mat W1, Mat b1, Mat W2, Mat b2 ){
      this.W1 = W1;
      this.b1 = b1;
      this.W2 = W2;
      this.b2 = b2;
    }
    /**
    * Generates a Neural Net with given specs.
    * @param {NetOpts} opt Specs of the Neural Net.
    */
    Net( NetOpts opt){
      this.initializeAsFreshInstance( opt );
    }

    Net(){
      // { architecture: { inputSize: 1, hiddenUnits: [1], outputSize: 1 } }
      NetOpts opt = new NetOpts();
      opt.architectureHiddenUnits = new int[1](1);
      opt.architectureInputSize = 1;
      opt.architectureOutputSize = 1;
      this.initializeAsFreshInstance( opt );
    }
/*
    Net(opt: any) {
      if (this.isFromJSON(opt)) {
        this.initializeFromJSONObject(opt);
      } else if (this.isFreshInstanceCall(opt)) {
        this.initializeAsFreshInstance(opt);
      } else {
        this.initializeAsFreshInstance({ architecture: { inputSize: 1, hiddenUnits: [1], outputSize: 1 } });
      }
    }

    private isFromJSON(opt: any) {
      return Net.has(opt, ['W1', 'b1', 'W2', 'b2']);
    }

    private isFreshInstanceCall(opt: NetOpts) {
      return Net.has(opt, ['architecture']) && Net.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
    }

    private initializeFromJSONObject(opt: { W1, b1, W2, b2 }) {
      this.W1 = Mat.fromJSON(opt['W1']);
      this.b1 = Mat.fromJSON(opt['b1']);
      this.W2 = Mat.fromJSON(opt['W2']);
      this.b2 = Mat.fromJSON(opt['b2']);
    }
*/
    private void initializeAsFreshInstance(NetOpts opt) {
      float mu = 0f;
      float std = 0.01f;

      //if(Net.has(opt, ['other'])) {
        mu = opt.otherMu ? opt.otherMu : mu;
        std = opt.otherStd ? opt.otherStd : std;
      //}
      int firstLayer = 0; // only consider the first layer => shallowness
      this.W1 = new RandMat(opt.ArchitectureHiddenUnits[firstLayer], opt.architectureInputSize, mu, std);
      this.b1 = new Mat(opt.architectureHiddenUnits[firstLayer], 1);
      this.W2 = new RandMat(opt.architectureOutputSize, opt.architectureHiddenUnits[firstLayer], mu, std);
      this.b2 = new Mat(opt.architectureOutputSize, 1);
    }

    /**
    * Updates all weights
    * @param alpha discount factor for weight updates
    */
    public void update(float alpha) {
      this.W1.update(alpha);
      this.b1.update(alpha);
      this.W2.update(alpha);
      this.b2.update(alpha);
    }

    public static string toJSON(Net net) {
      string json = "";
      /*
      json['W1'] = Mat.toJSON(net.W1);
      json['b1'] = Mat.toJSON(net.b1);
      json['W2'] = Mat.toJSON(net.W2);
      json['b2'] = Mat.toJSON(net.b2);*/
      return json;
    }

    /**
    * Compute forward pass of Neural Network
    * @param state 1D column vector with observations
    * @param graph optional: inject Graph to append Operations
    * @returns output of type `Mat`
    */
    public Mat forward(Mat state, Graph graph ) {
      Mat weightedInput = graph.mul(this.W1, state);

      Mat a1mat = graph.add(weightedInput, this.b1);

      Mat h1mat = graph.tanh(a1mat);

      Mat a2Mat = this.computeOutput(h1mat, graph);
      return a2Mat;
    }

    private Mat computeOutput(Mat hiddenUnits , Graph graph ) {
      Mat weightedActivation = graph.mul(this.W2, hiddenUnits);
      // a2 = Output Vector of Weight2 (W2) and hyperbolic Activation (h1)
      Mat a2Mat = graph.add(weightedActivation, this.b2);
      return a2Mat;
    }

    public static Net fromJSON(string json) { //: { W1, b1, W2, b2 }
    //json
      return new Net();
    }

/*
    private static has(obj: any, keys: Array<string>) {
      for (const key of keys) {
        if (Object.hasOwnProperty.call(obj, key)) { continue; }
        return false;
      }
      return true;
    }
*/
  }
}
