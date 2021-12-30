using System;
using System.Collections;


namespace Recurrent{

    class Utils {

        public static Random random = new Random();

        /**
        * Returns a random floating point number of a uniform distribution between `min` and `max`
        * @param {number} min lower bound
        * @param {number} max upper bound
        * @returns {number} random float value
        */
        public static float randf(float min, float max) {
            return random.NextSingle() * (max - min) + min;
        }

        /**
        * Returns a random integer number of a uniform distribution between [`min`, `max`)
        * @param {number} min lower bound
        * @param {number} max upper bound
        * @returns {number} random integer value
        */
        public static int randi(int min, int max) {
            return random.Next(min, max); // greater than or equal to minValue and less than maxValue; that is, the range of return values includes minValue but not maxValue
            //return MathF.Floor(Utils.randf(min, max));
        }

        /**
        * Returns a sample of a normal distribution
        * @param {number} mu mean
        * @param {number} std standard deviation
        * @returns {number} random value
        */
        public static float randn(float mu, float std) {
            return mu + Utils.gaussRandom() * std;
        }

        /**
        * Returns a random sample number from a normal distributed set
        * @param {number} min lower bound
        * @param {number} max upper bound
        * @param {number} skew factor of skewness; < 1 shifts to the right; > 1 shifts to the left
        * @returns {number} random value
        */
        public static float skewedRandn(float mu, float std, float skew) {
            float sample = Utils.box_muller();
            sample = MathF.Pow(sample, skew);
            sample = (sample - 0.5f) * 10f;
            return mu + sample * std;
        }

        /**
        * Gaussian-distributed sample from a normal distributed set.
        */
        private static float gaussRandom() {
            return (Utils.box_muller() - 0.5f) * 10f;
        }

        /**
        * Box-Muller Transform, to transform uniform random values into standard gaussian distributed random values.
        * @returns random value between of interval (0,1)
        */
        private static float box_muller() {
            // Based on:
            // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
            // and
            // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
            float z0 = 0f, u1 = 0f, u2 = 0f;
            do {
                u1 = u2 = 0f;
            // Convert interval from [0,1) to (0,1)
                do { u1 = random.NextSingle(); } while (u1 == 0f);
                do { u2 = random.NextSingle(); } while (u2 == 0f);
                z0 = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
                z0 = (z0 * 0.1f) + 0.5f;
            } while (z0 > 1f || z0 < 0f); // resample c
            return z0;
        }

        /**
        * Calculates the sum of a given set
        * @param arr randomly populated array of numbers
        */
        public static float sum(float[] arr) {
            float sum = 0;
            for (int i = 0; i < arr.length; i++) {
                sum += arr[i];
            }
            return sum;
        }

        /**
        * Calculates the mean of a given set
        * @param arr set of values
        */
        public static float mean(float[] arr) {
            // mean of [3, 5, 4, 4, 1, 1, 2, 3] is 2.875
            int count = arr.Length;
            float sum = Utils.sum(arr);
            return sum / count;
        }

        /**
        * Calculates the median of a given set
        * @param arr set of values
        */
        public static float median(float[] arr) {
            // median of [3, 5, 4, 4, 1, 1, 2, 3] = 3
            float median = 0;
            int count = arr.Length;
            Array.Sort(arr);
            if (count % 2 == 0) { // is even
            // average of two middle numbers
                median = (arr[count / 2 - 1] + arr[count / 2]) / 2;
            } else { // is odd
            // middle number only
                median = arr[(count - 1) / 2];
            }
            return median;
        }

        /**
        * Calculates the standard deviation of a given set
        * @param arr set of values
        * @param precision the floating point precision for grouping results, e.g. 1e3 [defaults to 1e6]
        * mph - TODO: C# conversion - this could be better, needs some time spent to rework, doesn't appear to be used currently though !
        */
        public static float[] mode(float[] arr, float precision = 1e6){
            // as result can be bimodal or multimodal,
            // the returned result is provided as an array
            // mode of [3, 5, 4, 4, 1, 1, 2, 3] = [1, 3, 4]
            //precision = precision ? precision : 1e6;
            List<float> modes = new List<float>();
            Dictionary<int,float> count = new Dictionary<int,float>();
            int num = 0;
            int maxCount = 0;
            // populate array with number counts
            for (int i = 0; i < arr.Length; i++) {
                num = (int)MathF.Round(arr[i] * precision) / precision;
                count[num] = (count[num] || 0) + 1; // initialize or increment for number
                if (count[num] > maxCount) {
                    maxCount = count[num]; // memorize count value of max index
                }
            }
            // memorize numbers equal with maxCount
            //for (int i in count) {
            foreach (var (key, value) in count) {
               // if (count.hasOwnProperty(i)) {
                    //if (count[i] === maxCount) {
                    if (value == maxCount) {    
                        modes.Add( (float)key );
                    }
                //}
            }
            return modes.ToArray();
        }

        /**
        * Calculates the population variance (uncorrected), the sample variance (unbiased) or biased variance of a given set
        * @param arr set of values
        * @param normalization defaults to sample variance ('unbiased') 
        *  'uncorrected' | 'biased' | 'unbiased'
        */
        public static float var(float[] arr, string normalization = "unbiased" ) {
            //normalization = normalization ? normalization : 'unbiased';
            int count = arr.Length;

            // calculate the variance
            float mean = Utils.mean(arr);
            float sum = 0;
            float diff = 0;

            for (int i = 0; i < count; i++) {
                diff = arr[i] - mean;
                sum += diff * diff;
            }

            switch (normalization) {
                case "uncorrected":
                    return (float)(sum / count);
                break;
                case "biased":
                    return (float)(sum / (count + 1));
                break;
                case "unbiased":
                    return (float)((count == 1) ? 0 : sum / (count - 1));
                break;
                default:
                    return ;
                }
        }

        /**
        * Calculates the standard deviation of a given set
        * @param arr set of values
        * @param normalization defaults to sample variance ('unbiased')
        */
        public static float std(float[] arr, string normalization = "unbiased") {
            return MathF.Sqrt( Utils.var(arr, normalization));
        }

        /**
        * Fills the given array with normal distributed random values.
        * @param arr Array to be filled
        * @param mu mean
        * @param std standard deviation
        * @returns {void} void
        */
        public static void fillRandn(ref float[] arr, float mu, float std) {
            for (int i = 0; i < arr.length; i++) { 
                arr[i] = Utils.randn(mu, std); 
            }
        }

        /**
        * Fills the given array with uniformly distributed random values between `min` and `max`.
        * @param arr Array to be filled
        * @param min lower bound
        * @param max upper bound
        * @returns {void} void
        */
        public static void fillRand(ref float[] arr, float min, float max){
            for (int i = 0; i < arr.length; i++) { 
                arr[i] = Utils.randf(min, max); 
            }
        }

        /**
        * Fills the pointed array with constant values.
        * @param {Array<number> | Float64Array} arr Array to be filled
        * @param {number} c value
        * @returns {void} void
        */
        public static void fillConst(ref float[] arr, float c) {
            for (int i = 0; i < arr.length; i++) { 
                arr[i] = c; 
            }
        }

        /**
        * returns array populated with ones of length n and uses typed arrays if available
        * @param {number} n length of Array
        * @returns {Array<number> | Float64Array} Array
        */
        public static float[] ones(int n){
            return Utils.fillArray(n, 1);
        }

        /**
        * returns array of zeros of length n and uses typed arrays if available
        * @param {number} n length of Array
        * @returns {Array<number> | Float64Array} Array
        */
        public static float[] zeros(int n) {
            return Utils.fillArray(n, 0);
        }

        private static float[] fillArray(int n, float val) {

            if ( n == null || Single.NaN == n ) { 
                return new float[]; 
            }

            float[] arr = new float[n]();
            for ( int i = 0; i < arr.Length;i++ ) {
                arr[i] = val;
            }
            return arr;
            
        }

        /**
        * Softmax of a given set of values
        * @param {Array<number> | Float64Array} arr set of values
        */
        public static float[] softmax(float[] arr) {
            float[] output = new float[]();
            float expSum = 0;
            for(int i = 0; i < arr.length; i++) {
                expSum += MathF.Exp(arr[i]);
            }
            //adding small optimisation - multiply rather than divide in possible large iteration
            float expSumMultiplier = 1f / expSum;
            for(let i = 0; i < arr.length; i++) {
                output[i] = MathF.Exp(arr[i]) * expSumMultiplier;
            }

            return output;
        }

        /**
        * Argmax of a given set of values
        * @param {Array<number> | Float64Array} arr set of values
        * @returns {number} Index of Argmax Operation
        */
        public static int argmax(float[] arr) {
            float maxValue = arr[0];
            int maxIndex = 0;
            for (int i = 1; i < arr.length; i++) {
                //const v = arr[i];
                if (arr[i] > maxValue) {
                    maxIndex = i;
                    maxValue = arr[i];
                }
            }
            return maxIndex;
        }

        /**
        * Returns an index of the weighted sample of Array `arr`
        * @param {Array<number> | Float64Array} arr Array to be sampled
        * @returns {number} 
        */
        public static int sampleWeighted(float[] arr) {
            float r = random.NextSingle(); //in js this is between 0 - 1
            float c = 0f;
            for (int i = 0; i < arr.length; i++) {
                c += arr[i];
                if (c >= r) { 
                    return i; 
                }
            }

            return 0;
        }

    }



}