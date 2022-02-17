//import { Mat } from '.';
//import { MatOps } from './utils/mat-ops';
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Recurrent{

  public class Graph {

    private bool needsBackpropagation;

    private readonly List<GraphStack> backpropagationStack; //adding enums here and then will have a case statement in the processing


    private Mat m1;
    private Mat m2;
    private Mat mOut;

    /**
    * Initializes a Graph to memorize Matrix Operation Sequences.
    */
    public Graph() {
      this.needsBackpropagation = false;
      this.backpropagationStack = new List<GraphStack>();
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
        procesBackpropagationStack(this.backpropagationStack[i]);
      }
    }


    // Note: maybe a better architecture to move the MatOps into the GraphStack class, encapsulate all the properties and methods and then call a processing method to run through them
    // doing it like below is a bit too much pass the parcel...
    // Copying in original MatOps methods for comparison when it breaks - I will remove these in a few commits - once working
    private void procesBackpropagationStack(GraphStack entry){
      switch(entry.operation){

        case GraphOperations.RowPluck:

          //MatOps.getRowPluckBackprop( ref entry.m1, entry.rowIndex, ref entry.matOut);//ref Mat m, int rowIndex, Mat matOut );
          
          for (int i = 0; i < entry.m1.cols; i++) {
            entry.m1.dw[entry.m1.cols * entry.rowIndex + i] += entry.matOut.dw[i];
          }
          

          /*
          public static void getRowPluckBackprop(ref Mat m, int rowIndex, ref Mat matOut ){

                  for (int i = 0; i < m.cols; i++) {
                    m.dw[m.cols * rowIndex + i] += matOut.dw[i];
                  }

              }
          */

        break;
      
        case GraphOperations.Tanh:

          //MatOps.getTanhBackprop(ref entry.m1, ref entry.matOut);
          for (int i = 0; i < entry.m1.w.Length; i++) {
            // grad for z = tanh(x) is (1 - z^2)
            float mwi = entry.matOut.w[i];
            entry.m1.dw[i] += (1.0f - mwi * mwi) * entry.matOut.dw[i];
          }
          /*
          public static void getTanhBackprop(ref Mat m, ref Mat matOut) {
          
                  for (int i = 0; i < m.w.Length; i++) {
                    // grad for z = tanh(x) is (1 - z^2)
                    float mwi = matOut.w[i];
                    m.dw[i] += (1.0f - mwi * mwi) * matOut.dw[i];
                  }

              }
          */

        break;
        case GraphOperations.Sigmoid:

          //MatOps.getSigmoidBackprop(ref entry.m1, ref entry.matOut);
          for (int i = 0; i < entry.m1.w.Length; i++) {
            // grad for z = tanh(x) is (1 - z^2)
            float mwi = entry.matOut.w[i];
            entry.m1.dw[i] += mwi * (1.0f - mwi) * entry.matOut.dw[i];
          }

        /*
        public static void getSigmoidBackprop(ref Mat m, ref Mat matOut) {

        for (int i = 0; i < m.w.Length; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          float mwi = matOut.w[i];
          m.dw[i] += mwi * (1.0f - mwi) * matOut.dw[i];
        }

        }*/

        break;
        case GraphOperations.Relu:

          //MatOps.getReluBackprop(ref entry.m1, ref entry.matOut);

          for (int i = 0; i < entry.m1.w.Length; i++) {
            entry.m1.dw[i] += (entry.m1.w[i] > 0f) ? entry.matOut.dw[i] : 0.0f;
          }

          /*public static void getReluBackprop(ref Mat m, ref Mat matOut) {

              for (int i = 0; i < m.w.Length; i++) {
                m.dw[i] += (m.w[i] > 0f) ? matOut.dw[i] : 0.0f;
              }

          }*/

        break;
        case GraphOperations.Addition:

          //MatOps.getAddBackprop(ref entry.m1, ref entry.m2, ref entry.matOut);
          for (int i = 0; i < entry.m1.w.Length; i++) {
            entry.m1.dw[i] += entry.matOut.dw[i];
            entry.m2.dw[i] += entry.matOut.dw[i];
          }

          /*public static void getAddBackprop(ref Mat m1, ref Mat m2, ref Mat matOut) {
      
              for (int i = 0; i < m1.w.Length; i++) {
                m1.dw[i] += matOut.dw[i];
                m2.dw[i] += matOut.dw[i];
              }

          }*/

        break;
        case GraphOperations.Multiply:

          //MatOps.getMulBackprop(ref entry.m1, ref entry.m2, ref entry.matOut);
          if( entry.m1 == null || entry.m2 == null ){
            break;
          }


          for (int i = 0; i < entry.m1.rows; i++) {
            for (int j = 0; j < entry.m2.cols; j++) {
              for (int k = 0; k < entry.m1.cols; k++) {
                float b = entry.matOut.dw[entry.m2.cols * i + j];

              //  int ind1 = entry.m1.cols * i + k;
              //  int ind2 = entry.m2.cols * k + j;
              //Debug.Log("indexes" + ind1 + " " + ind2 );
              //  Debug.Log( entry.m1.dw.Length );
              //  Debug.Log( entry.m2.w.Length );
               // Debug.Log( entry.m1.hasData());
                //Debug.Log( entry.m2.hasData() );
                entry.m1.dw[entry.m1.cols * i + k] += entry.m2.w[entry.m2.cols * k + j] * b;
                entry.m2.dw[entry.m2.cols * k + j] += entry.m1.w[entry.m1.cols * i + k] * b;
              }
            }
          }

          /*public static void getMulBackprop(ref Mat m1, ref Mat m2, ref Mat matOut) {

              for (int i = 0; i < m1.rows; i++) {
                for (int j = 0; j < m2.cols; j++) {
                  for (int k = 0; k < m1.cols; k++) {
                    float b = matOut.dw[m2.cols * i + j];
                    m1.dw[m1.cols * i + k] += m2.w[m2.cols * k + j] * b;
                    m2.dw[m2.cols * k + j] += m1.w[m1.cols * i + k] * b;
                  }
                }
              }

          }*/

        break;
        case GraphOperations.Dot:

          //MatOps.getDotBackprop(ref entry.m1, ref entry.m2, ref entry.matOut);

          for (int i = 0; i < entry.m1.w.Length; i++) {
            entry.m1.dw[i] += entry.m2.w[i] * entry.matOut.dw[0];
            entry.m2.dw[i] += entry.m1.w[i] * entry.matOut.dw[0];
          }

          /*public static void getDotBackprop(ref Mat m1 ,ref Mat m2, ref Mat matOut) {

              for (int i = 0; i < m1.w.Length; i++) {
                m1.dw[i] += m2.w[i] * matOut.dw[0];
                m2.dw[i] += m1.w[i] * matOut.dw[0];
              }

          }*/

        break;
        case GraphOperations.ElementMultiply:

          //MatOps.getEltmulBackprop(ref entry.m1, ref entry.m2, ref entry.matOut);
          for (int i = 0; i < entry.m1.w.Length; i++) {
            entry.m1.dw[i] += entry.m2.w[i] * entry.matOut.dw[i];
            entry.m2.dw[i] += entry.m1.w[i] * entry.matOut.dw[i];
          }

           /*public static void getEltmulBackprop(ref Mat m1, ref Mat m2, ref Mat matOut) {

              for (int i = 0; i < m1.w.Length; i++) {
                m1.dw[i] += m2.w[i] * matOut.dw[i];
                m2.dw[i] += m1.w[i] * matOut.dw[i];
              }

          }*/
          
        break;
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
        //getRowPluckBackprop(ref Mat m, int rowIndex, Mat matOut ){
        //Delegate backward = MatOps.getRowPluckBackprop(m, rowIndex, matOut);
        GraphStack entry = new GraphStack(GraphOperations.RowPluck,  m, rowIndex,  matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getTanhBackprop(m, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Tanh,  m,  matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getSigmoidBackprop(m, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Sigmoid,  m,  matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getReluBackprop(m, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Relu, m, matOut );
        this.backpropagationStack.Add( entry );
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

    private void addAdditionToBackpropagationStack(Mat m1, Mat m2, Mat matOut) {
      if (this.needsBackpropagation) {
        //Delegate backward = MatOps.getAddBackprop(m1, m2, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Addition, m1, m2, matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getMulBackprop(m1, m2, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Multiply, m1, m2, matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getDotBackprop(m1, m2, matOut);
        GraphStack entry = new GraphStack(GraphOperations.Dot,  m1,  m2,  matOut );
        this.backpropagationStack.Add( entry );
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
        //Delegate backward = MatOps.getEltmulBackprop(m1, m2, matOut);
        GraphStack entry = new GraphStack(GraphOperations.ElementMultiply,  m1,  m2,  matOut );
        this.backpropagationStack.Add( entry );
      }
    }
  }

}