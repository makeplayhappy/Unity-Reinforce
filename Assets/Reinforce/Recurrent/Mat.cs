using System.Diagnostics;
namespace Recurrent{

    public class Mat {
    
        public readonly int rows;
        public readonly int cols;
        private readonly int _length;  // length of 1d-representation of Mat

        public float[] w;
        public float[] dw;

        /**
        * 
        * @param rows rows of Matrix
        * @param cols columns of Matrix
        */
        public Mat(int rowsIn, int colsIn) {

            System.Diagnostics.Debug.Assert(rowsIn <= 0 || colsIn <= 0, "[class:Mat] constructor: zero setting row ("+rowsIn+") col ("+colsIn+")");

            //super();
            rows = rowsIn;
            cols = colsIn;
            this._length = rows * cols;
            this.w = Utils.zeros(this._length);
            this.dw = Utils.zeros(this._length);// + cols);

//m1 dw [ cols * rows + cols ]
// md dw [ cols * m1.cols + cols]
        }

        public bool hasData(){
            if(rows > 0 && cols > 0 && w.Length > 0 ){
                return true;
            }
            return false;
        }

        /**
        * Accesses the value of given row and column.
        * @param row 
        * @param col
        * @returns the value of given row and column
        */
        public float get(int row, int col) {
            int ix = this.getIndexBy(row, col);
            //Mat.assert(ix >= 0 && ix < this.w.Length, '[class:Mat] get: index out of bounds.');
            System.Diagnostics.Debug.Assert(ix >= 0 && ix < this.w.Length, "[class:Mat] get: index out of bounds.");
            return this.w[ix];
        }

        /**
        * Mutates the value of given row and column.
        * @param row 
        * @param col 
        * @param v 
        */
        public void set(int row, int col, float v) {
            int ix = this.getIndexBy(row, col);
            //Mat.assert(ix >= 0 && ix < this.w.Length, '[class:Mat] set: index out of bounds.');
            System.Diagnostics.Debug.Assert(ix >= 0 && ix < this.w.Length, "[class:Mat] set: index out of bounds.");
            this.w[ix] = v;
        }

        /**
        * Gets Index by Row-major order
        * @param row 
        * @param col 
        */
        protected int getIndexBy(int row, int col) {
            return (row * this.cols) + col;
        }

        /**
        * Sets values according to the given Array.
        * @param arr 
        */
        public void setFrom(float[] arr) {
            for (int i = 0; i < arr.Length; i++) {
                this.w[i] = arr[i];
            }
        }

        /**
        * Overrides the values from the column of the matrix
        * @param m 
        * @param colIndex 
        */
        public void setColumn(Mat m, int colIndex) {
            System.Diagnostics.Debug.Assert(m.w.Length == this.rows, "[class:Mat] setColumn: dimensions misaligned.");
            //Mat.assert(m.w.Length === this.rows, '[class:Mat] setColumn: dimensions misaligned.');
            for (int i = 0; i < m.w.Length; i++) {
                this.w[(this.cols * i) + colIndex] = m.w[i];
            }
        }

        /**
        * Checks equality of matrices.
        * The check includes the value equality and a dimensionality check.
        * Derivatives are not considered.
        * @param {Mat} m Matrix to be compared with
        * @returns {boolean} true if equal and false otherwise
        */
        public bool equals(Mat m) {
            if(this.rows != m.rows || this.cols != m.cols) {
                return false;
            }
            for(int i = 0; i < this._length; i++) {
                if(this.w[i] != m.w[i]) {
                    return false;
                }
            }
            return true;
        }

        // mph - JSONS TODO
        public static string toJSON(Mat m) { // : {rows, cols, w}
            string json = "";//{rows: 0, cols: 0, w: []};
            //json.rows = m.rows || m.n;
            //json.cols = m.cols || m.d;
            //json.w = m.w;
            return json;
        }

        public static Mat fromJSON(string json) { // : {rows, n?, cols, d?, w}
            //const rows = json.rows || json.n;
            //const cols = json.cols || json.d;
            int rowsInit = 1;
            int colsInit = 1;
            Mat mat = new Mat(rowsInit, colsInit);
            //for (let i = 0; i < mat._length; i++) {
            //    mat.w[i] = json.w[i];
            //}
            return mat;
        }

        /**
        * Discounts all values as follows: w[i] = w[i] - (alpha * dw[i])
        * @param alpha discount factor
        */
        public void update(float alpha) {
            for (int i = 0; i < this._length; i++) {
                if (this.dw[i] != 0) {
                    this.w[i] = this.w[i] - alpha * this.dw[i];
                    this.dw[i] = 0f;
                }
            }
        }

    }
}
