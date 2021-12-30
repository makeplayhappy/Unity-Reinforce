using Recurrent;

namespace Reinforce{
    public class SarsaExperience
    {
        Mat s0;      // last state after acting (from t-1)
        int a0;   // last action Index after acting (from t-1)
        float r0;   // current reward after learning (from t)
        Mat s1;      // current state while acting (from t)
        int a1;   // current action Index while acting (from t)
    }

    public SarsaExperience(){
        this.s0 = null;
        this.a0 = null;
        this.r0 = null;
        this.s1 = null;
        this.a1 = null;
    }

    public SarsaExperience(Mat s0, int a0, float r0, Mat s1, int a1){
        this.s0 = s0;
        this.a0 = a0;
        this.r0 = r0;
        this.s1 = s1;
        this.a1 = a1;


    }


}