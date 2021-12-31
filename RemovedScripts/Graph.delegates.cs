//import { Mat } from '.';
//import { MatOps } from './utils/mat-ops';
using System;
using System.Collections.Generic;

namespace Recurrent{
  public class Graph {

    private bool needsBackpropagation;

    private readonly List<Delegate> backpropagationStack ;

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
    public void memorizeOperationSequence(bool isMemorizing = false) {
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
      for (int i = this.backpropagationStack.Count - 1; i >= 0; i--) {
        this.backpropagationStack[i]();
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
        Delegate backward = MatOps.getRowPluckBackprop(m, rowIndex, matOut);
        this.backpropagationStack.Add(backward);
      }
    }

    /**
    * Non-destructively pluck a row of m with rowIndex
    * @param m 
    * @param rowIndex 
    */
    public Mat gauss(Mat m, Mat std) {
      Mat matOut = MatOps.gauss(m, std);
      return matOut;
    }

    /**
    * Non-destructive elementwise tanh
    * @param m 
    */
    public Mat tanh(Mat m) {
      Mat matOut = MatOps.tanh(m);
      this.addTanhToBackpropagationStack(m, matOut);
      return matOut;
    }

    private void addTanhToBackpropagationStack(Mat m ,Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getTanhBackprop(m, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise sigmoid
    * @param m 
    */
    public Mat sig(Mat m) {
      Mat matOut = MatOps.sig(m);
      this.addSigmoidToBackpropagationStack(m, matOut);
      return matOut;
    }

    private void addSigmoidToBackpropagationStack(Mat m, Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getSigmoidBackprop(m, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise ReLU (rectified linear unit)
    * @param m 
    */
    public Mat relu(Mat m) {
      Mat matOut = MatOps.relu(m);
      this.addReluToBackpropagationStack(m, matOut);
      return matOut;
    }

    private void addReluToBackpropagationStack(Mat m, Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getReluBackprop(m, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive elementwise addition
    * @param m1 
    * @param m2 
    */
    public Mat add(Mat m1, Mat m2) {
      Mat matOut = MatOps.add(m1, m2);
      this.addAdditionToBackpropagationStack(m1, m2, matOut);
      return matOut;
    }

    private void addAdditionToBackpropagationStack(Mat m1, Mat m2,Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getAddBackprop(m1, m2, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive matrix multiplication
    * @param m1 
    * @param m2 
    */
    public Mat mul(Mat m1, Mat m2) {
      Mat matOut = MatOps.mul(m1, m2);
      this.addMultiplyToBackpropagationStack(m1, m2, matOut);
      return matOut;
    }

    private void addMultiplyToBackpropagationStack(Mat m1, Mat m2, Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getMulBackprop(m1, m2, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructive Dot product.
    * @param m1 
    * @param m2 
    */
    public Mat dot(Mat m1, Mat m2) {
      Mat matOut = MatOps.dot(m1, m2);
      this.addDotToBackpropagationStack(m1, m2, matOut);
      return matOut;
    }

    private void addDotToBackpropagationStack(Mat m1, Mat m2, Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getDotBackprop(m1, m2, matOut);
        this.backpropagationStack.push(backward);
      }
    }

    /**
    * Non-destructively elementwise multiplication
    * @param m1 
    * @param m2 
    */
    public Mat eltmul(Mat m1, Mat m2) {
      Mat matOut = MatOps.eltmul(m1, m2);
      this.addEltmulToBackpropagationStack(m1, m2, matOut);
      return matOut;
    }

    private void addEltmulToBackpropagationStack(Mat m1, Mat m2, Mat matOut) {
      if (this.needsBackpropagation) {
        Delegate backward = MatOps.getEltmulBackprop(m1, m2, matOut);
        this.backpropagationStack.push(backward);
      }
    }
  }

}