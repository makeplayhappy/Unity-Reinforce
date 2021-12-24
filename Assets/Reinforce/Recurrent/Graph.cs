//import { Mat } from '.';
//import { MatOps } from './utils/mat-ops';
using System;
namespace Recurrent{
  class Graph {

    private bool needsBackpropagation;

    private readonly IList<Delegate> backpropagationStack ;

    /**
    * Initializes a Graph to memorize Matrix Operation Sequences.
    */
    Graph() {
      this.needsBackpropagation = false;
      this.backpropagationStack = new List<Delegate>();
    }

    /**
    * Switch whether to memorize the operation sequence for Backpropagation (true) or ignore it (false).
    * @param {boolean} isMemorizing true or false [defaults to false]
    */
    public void memorizeOperationSequence(boolean isMemorizing = false) {
      this.needsBackpropagation = isMemorizing;
    }

    /**
    * Gives back the state of either memorizing or not a sequence of operations
    */
    public bool isMemorizingSequence() {
      return this.needsBackpropagation;
    }

    /**
    * Clears the memorized sequence of operations
    */
    public void forgetCurrentSequence() {
      this.backpropagationStack.Clear(); // reset list
    }

    /**
    * Executes the memorized sequence of derivative operations in LIFO order (last in first out)
    */
    public void backward() {
      for (int i = this.backpropagationStack.length - 1; i >= 0; i--) {
        this.backpropagationStack[i].Invoke();
      }
    }

    /**
    * Non-destructively pluck a row of m with rowIndex
    * @param m 
    * @param rowIndex 
    */
    public Mat rowPluck(Mat m, int rowIndex) {
      Mat matOut = MatOps.rowPluck(m, rowIndex);
      this.addRowPluckToBackpropagationStack(m, rowIndex, matOut);
      return matOut;
    }

    private void addRowPluckToBackpropagationStack(Mat m, int rowIndex, Mat matOut ) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getRowPluckBackprop(m, rowIndex, matOut);
        this.backpropagationStack.Add(backward);
      }
    }

    /**
    * Non-destructively pluck a row of m with rowIndex
    * @param m 
    * @param rowIndex 
    */
    public gauss(m: Mat, std: Mat): Mat {
      const out = MatOps.gauss(m, std);
      return out;
    }

    /**
    * Non-destructive elementwise tanh
    * @param m 
    */
    public tanh(m: Mat): Mat {
      const out = MatOps.tanh(m);
      this.addTanhToBackpropagationStack(m, out);
      return out;
    }

    private addTanhToBackpropagationStack(m: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getTanhBackprop(m, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise sigmoid
    * @param m 
    */
    public sig(m: Mat): Mat {
      const out = MatOps.sig(m);
      this.addSigmoidToBackpropagationStack(m, out);
      return out;
    }

    private addSigmoidToBackpropagationStack(m: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getSigmoidBackprop(m, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise ReLU (rectified linear unit)
    * @param m 
    */
    public relu(m: Mat): Mat {
      const out = MatOps.relu(m);
      this.addReluToBackpropagationStack(m, out);
      return out;
    }

    private addReluToBackpropagationStack(m: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getReluBackprop(m, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise addition
    * @param m1 
    * @param m2 
    */
    public add(m1: Mat, m2: Mat): Mat {
      const out = MatOps.add(m1, m2);
      this.addAdditionToBackpropagationStack(m1, m2, out);
      return out;
    }

    private addAdditionToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getAddBackprop(m1, m2, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive matrix multiplication
    * @param m1 
    * @param m2 
    */
    public mul(m1: Mat, m2: Mat): Mat {
      const out = MatOps.mul(m1, m2);
      this.addMultiplyToBackpropagationStack(m1, m2, out);
      return out;
    }

    private addMultiplyToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getMulBackprop(m1, m2, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive Dot product.
    * @param m1 
    * @param m2 
    */
    public dot(m1: Mat, m2: Mat): Mat {
      const out = MatOps.dot(m1, m2);
      this.addDotToBackpropagationStack(m1, m2, out);
      return out;
    }

    private addDotToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getDotBackprop(m1, m2, out);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructively elementwise multiplication
    * @param m1 
    * @param m2 
    */
    public eltmul(m1: Mat, m2: Mat): Mat {
      const out = MatOps.eltmul(m1, m2);
      this.addEltmulToBackpropagationStack(m1, m2, out);
      return out;
    }

    private addEltmulToBackpropagationStack(m1: Mat, m2: Mat, out: Mat) {
      if (this.needsBackpropagation) {
        const backward = MatOps.getEltmulBackprop(m1, m2, out);
        this.backpropagationStack.push(backward);
      }
    }
  }

}