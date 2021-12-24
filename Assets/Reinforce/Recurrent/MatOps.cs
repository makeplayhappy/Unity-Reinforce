using System.Diagnostics;
namespace Recurrent{
  class MatOps {

    /**
    * Non-destructively pluck a row of m with rowIndex
    * @param m 
    * @param rowIndex index of row
    * @returns a column Vector [cols, 1]
    */
    public static Mat rowPluck(Mat m , int rowIndex) {
      Debug.Assert(rowIndex >= 0 && rowIndex < m.rows, "[class:MatOps] rowPluck: dimensions misaligned");
      Mat matOut = new Mat(m.cols, 1);
      for (int i = 0; i < m.cols; i++) {
        matOut.w[i] = m.w[m.cols * rowIndex + i];
      }
      return matOut;
    }


    /**
    * Need to pass by reference into an Action ... C# doesnt really have a good solution - maybe delegates work
    * This may need complete re architecting
    */
    delegate void refMatIntMat(ref Mat m, int rowIndex, Mat matOut);


    public static refMatIntMat getRowPluckBackprop(ref Mat m, int rowIndex, Mat matOut ){
      return () => {
        for (int i = 0; i < m.cols; i++) {
          m.dw[m.cols * rowIndex + i] += matOut.dw[i];
        }
      };
    }
    /* the JS implementation * /
    public static getRowPluckBackprop(m: Mat, rowIndex: number, out: Mat): Function {
      return () => {
        for (let i = 0; i < m.cols; i++) {
          m.dw[m.cols * rowIndex + i] += out.dw[i];
        }
      };
    }
    /* */
    /*
    Action<MyComplexObject> myAction = (MyComplexObject result) =>
{
    result.Value = MyMethodThatReturnsSomething();                                              
};

*/

    /**
    * Non-destructive elementwise gaussian-distributed noise-addition.
    * @param {Mat} m 
    * @param {number} std Matrix with STD values
    * @returns {Mat} Matrix with results
    */
    public static Mat gauss(Mat m, Mat std) {
      //Mat.assert(m.w.length === std.w.length, '[class:MatOps] gauss: dimensions misaligned');
      Debug.Assert(m.w.length == std.w.length, "[class:MatOps] gauss: dimensions misaligned");
      Mat matOut = new Mat(m.rows, m.cols);
      for (int i = 0; i < m.w.length; i++) {
        matOut.w[i] = Utils.randn(m.w[i], std.w[i]);
      }
      return matOut;
    }

    /**
    * Non-destructive elementwise tanh.
    * @param {Mat} m
    * @returns {Mat} Matrix with results
    */
    public static Mat tanh(Mat m) {
      Mat matOut = new Mat(m.rows, m.cols);
      for (int i = 0; i < m.w.length; i++) {
        matOut.w[i] = MathF.Tanh(m.w[i]);
      }
      return matOut;
    }


    delegate void refMatMat(ref Mat m, Mat matOut);

    public static refMatMat getTanhBackprop(ref Mat m, Mat matOut) {
      return () => {
        for (int i = 0; i < m.w.length; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          float mwi = matOut.w[i];
          m.dw[i] += (1.0f - mwi * mwi) * matOut.dw[i];
        }
      };
    }

    /**
    * Non-destructive elementwise sigmoid.
    * @param m 
    * @returns Mat with results
    */
    public static Mat sig(Mat m) {
      Mat matOut = new Mat(m.rows, m.cols);
      for (int i = 0; i < m.w.length; i++) {
        matOut.w[i] = MatOps.sigmoid(m.w[i]);
      }
      return matOut;
    }
    
    private static float sigmoid(float x) {
      // helper function for computing sigmoid
      return 1.0f / (1f + MathF.Exp(-x));
    }

    public static refMatMat getSigmoidBackprop(ref Mat m, Mat matOut) {
      return () => {
        for (int i = 0; i < m.w.length; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          float mwi = matOut.w[i];
          m.dw[i] += mwi * (1.0f - mwi) * matOut.dw[i];
        }
      };
    }

    /**
    * Non-destructive elementwise ReLu.
    * @returns Mat with results
    */
    public static Mat relu(Mat m) {
      Mat matOut = new Mat(m.rows, m.cols);
      for (int i = 0; i < m.w.length; i++) {
        matOut.w[i] = MathF.Max(0, m.w[i]); // relu
      }
      return matOut;
    }

    public static refMatMat getReluBackprop(ref Mat m, Mat matOut) {
      return () => {
        for (int i = 0; i < m.w.length; i++) {
          m.dw[i] += m.w[i] > 0 ? matOut.dw[i] : 0.0;
        }
      };
    }

    /**
    * Non-destructive elementwise add.
    * @param {Mat} m1 
    * @param {Mat} m2 
    */
    public static Mat add(Mat m1, Mat m2) {
      //Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] add: dimensions misaligned');
      Debug.Assert(m1.w.length == m2.w.length && m1.rows == m2.rows, "[class:MatOps] add: dimensions misaligned");
      Mat matOut = new Mat(m1.rows, m1.cols);
      for (int i = 0; i < m1.w.length; i++) {
        matOut.w[i] = m1.w[i] + m2.w[i];
      }
      return matOut;
    }

    delegate void refMatRefMatMat(ref Mat m1, ref Mat m2, Mat matOut);
    public static refMatRefMatMat getAddBackprop(Mat m1, Mat m2, Mat matOut) {
      return () => {
        for (int i = 0; i < m1.w.length; i++) {
          m1.dw[i] += matOut.dw[i];
          m2.dw[i] += matOut.dw[i];
        }
      };
    }

    /**
    * Non-destructive Matrix multiplication.
    * @param m1 
    * @param m2 
    * @returns Mat with results
    */
    public static Mat mul(Mat m1, Mat m2) {
      //Mat.assert(m1.cols === m2.rows, '[class:MatOps] mul: dimensions misaligned');
      Debug.Assert(m1.cols == m2.rows, "[class:MatOps] mul: dimensions misaligned");
      Mat matOut = new Mat(m1.rows, m2.cols);
      for (int row = 0; row < m1.rows; row++) { // loop over rows of m1
        for (int col = 0; col < m2.cols; col++) { // loop over cols of m2
          float dot = 0.0;
          for (int k = 0; k < m1.cols; k++) { // dot product loop
            dot += m1.w[m1.cols * row + k] * m2.w[m2.cols * k + col];
          }
          matOut.w[m2.cols * row + col] = dot;
        }
      }
      return matOut;
    }

    public static refMatRefMatMat getMulBackprop(ref Mat m1, ref Mat m2, Mat matOut) {
      return () => {
        for (int i = 0; i < m1.rows; i++) {
          for (int j = 0; j < m2.cols; j++) {
            for (int k = 0; k < m1.cols; k++) {
              float b = matOut.dw[m2.cols * i + j];
              m1.dw[m1.cols * i + k] += m2.w[m2.cols * k + j] * b;
              m2.dw[m2.cols * k + j] += m1.w[m1.cols * i + k] * b;
            }
          }
        }
      };
    }

    /**
    * Non-destructive dot Product.
    * @param m1 
    * @param m2 
    * @return {Mat} Matrix of dimension 1x1
    */
    public static Mat dot(Mat m1 ,Mat m2)  {
      //Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] dot: dimensions misaligned');
      Debug.Assert(m1.w.length == m2.w.length && m1.rows == m2.rows, "[class:MatOps] dot: dimensions misaligned");
      Mat matOut = new Mat(1, 1);
      float dot = 0.0f;
      for (int i = 0; i < m1.w.length; i++) {
        dot += m1.w[i] * m2.w[i];
      }
      matOut.w[0] = dot;
      return matOut;
    }

    public static refMatRefMatMat getDotBackprop(ref Mat m1 ,ref Mat m2, Mat matOut) {
      return () => {
        for (int i = 0; i < m1.w.length; i++) {
          m1.dw[i] += m2.w[i] * matOut.dw[0];
          m2.dw[i] += m1.w[i] * matOut.dw[0];
        }
      };
    }

    /**
    * Non-destructive elementwise Matrix multiplication.
    * @param m1 
    * @param m2 
    * @return {Mat} Matrix with results
    */
    public static Mat eltmul(Mat m1, Mat m2) {
      //Mat.assert(m1.w.length === m2.w.length && m1.rows === m2.rows, '[class:MatOps] eltmul: dimensions misaligned');
      Debug.Assert(m1.w.length == m2.w.length && m1.rows == m2.rows, "[class:MatOps] eltmul: dimensions misaligned");
      Mat matOut = new Mat(m1.rows, m1.cols);
      for (int i = 0; i < m1.w.length; i++) {
        matOut.w[i] = m1.w[i] * m2.w[i];
      }
      return matOut;
    }
    delegate void refMatRefMatMat(ref Mat m1, ref Mat m2, Mat matOut);
    public static refMatRefMatMat getEltmulBackprop(ref Mat m1, ref Mat m2, Mat matOut) {
      return () => {
        for (int i = 0; i < m1.w.length; i++) {
          m1.dw[i] += m2.w[i] * matOut.dw[i];
          m2.dw[i] += m1.w[i] * matOut.dw[i];
        }
      };
    }
  }
}