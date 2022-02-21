using Recurrent;

namespace Reinforce{
    public class SarsaExperience
    {
        public Mat s0;      // last state after acting (from t-1)
        public int? a0;   // last action Index after acting (from t-1)
        public float? r0;   // current reward after learning (from t)
        public Mat s1;      // current state while acting (from t)
        public int? a1;   // current action Index while acting (from t)
    

        public SarsaExperience(){
            this.s0 = null;
            this.a0 = null;
            this.r0 = null;
            this.s1 = null;
            this.a1 = null;
        }

        public SarsaExperience(Mat in_s0, int? in_a0, float? in_r0, Mat in_s1, int? in_a1){
            this.s0 = in_s0;
            this.a0 = (in_a0 == null) ? 0 : (int)in_a0;
            this.r0 = (in_r0 == null) ? 0f : (float)in_r0;
            this.s1 = in_s1;
            this.a1 = (in_a1 == null) ? 0 : (int)in_a1;


        }
    }


}