using Recurrent;

namespace Reinforce{
    public interface SarsaExperience
    {
        Mat s0;      // last state after acting (from t-1)
        int a0;   // last action Index after acting (from t-1)
        float r0;   // current reward after learning (from t)
        Mat s1;      // current state while acting (from t)
        int a1;   // current action Index while acting (from t)
    }
}