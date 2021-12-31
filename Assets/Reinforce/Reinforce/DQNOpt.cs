//import { Opt } from './../.';

namespace Reinforce{
  public class DQNOpt {
    public bool trainingMode = true;
    public int[] numberOfHiddenUnits = new int[100];
    public float epsilonMax = 1.0f;
    public float epsilonMin = 0.1f;
    public int epsilonDecayPeriod = 1000000;// 1e6;
    public float epsilon = 0.05f;
    
    public float gamma = 0.9f;
    public float alpha = 0.01f;
    public int experienceSize = 1000000;//1e6;
    public bool doLossClipping = true;
    public float lossClamp = 1.0f;
    public bool doRewardClipping = true;
    public float rewardClamp = 1.0f;
    
    public int keepExperienceInterval = 25;
    public int replaySteps = 10;

    /**
    * Sets the number of neurons in hidden layer (currently only 1-d layer accepted)
    * @param numberOfHiddenUnits defaults to [ 100 ]
    */
    public void setNumberOfHiddenUnits(int[] numberOfHiddenUnits) {
      // TODO: Add DNN support
      this.numberOfHiddenUnits = numberOfHiddenUnits;
    }

    /**
    * Defines a linear annealing of Epsilon for an Epsilon Greedy Policy during 'training' = true
    * @param epsilonMax upper bound of epsilon; defaults to 1.0
    * @param epsilonMin lower bound of epsilon; defaults to 0.1
    * @param epsilonDecayPeriod number of timesteps; defaults to 1e6f
    */
    public void setEpsilonDecay(float epsilonMax, float epsilonMin, int epsilonDecayPeriod) {
      this.epsilonMax = epsilonMax;
      this.epsilonMin = epsilonMin;
      this.epsilonDecayPeriod = epsilonDecayPeriod;
    }

    /**
    * Sets the Epsilon Factor (Exploration Factor or Greedy Policy) during 'training' = false
    * @param epsilon value from [0,1); defaults to 0.05
    */
    public void setEpsilon(float epsilon) {
      this.epsilon = epsilon;
    }

    /**
    * Sets the Future Reward Discount Factor
    * @param gamma value from [0,1); defaults to 0.9
    */
    public void setGamma(float gamma) {
      this.gamma = gamma;
    }

    /**
    * Sets the Value Function Learning Rate
    * @param alpha defaults to 0.01
    */
    public void setAlpha(float alpha) {
      this.alpha = alpha;
    }

    /**
    * Activates or deactivates the Reward clipping to -1 or +1.
    * @param doLossClipping defaults to true (active)
    */
    public void setLossClipping(bool doLossClipping) {
      this.doLossClipping = doLossClipping;
    }

    /**
    * Sets the loss clamp for robustness
    * @param lossClamp defaults to 1.0
    */
    public void setLossClamp(float lossClamp) {
      this.lossClamp = lossClamp;
    }

    /**
    * Activates or deactivates the Reward clipping to -1 or +1.
    * @param doRewardClipping defaults to true (active)
    */
    public void setRewardClipping(bool doRewardClipping) {
      this.doRewardClipping = doRewardClipping;
    }

    /**
    * Activates or deactivates the Reward clipping to -1 or +1.
    * @param rewardClamp defaults to 1.0
    */
    public void setRewardClamp(float rewardClamp) {
      this.rewardClamp = rewardClamp;
    }

    /**
    * Activates or deactivated the Training Mode of the Solver.
    * @param trainingMode defaults to true (active)
    */
    public void setTrainingMode(bool trainingMode) {
      this.trainingMode = trainingMode;
    }

    /**
    * Sets Replay Memory Size
    * @param experienceSize defaults to 1e6f
    */
    public void setExperienceSize(int experienceSize) {
      this.experienceSize = experienceSize;
    }

    /**
    * Sets the amount of time steps before another experience is added to replay memory
    * @param keepExperienceInterval defaults to 25
    */
    public void setReplayInterval(int keepExperienceInterval) {
      this.keepExperienceInterval = keepExperienceInterval;
    }

    /**
    * Sets the amount of memory replays per iteration
    * @param replaySteps defaults to 10
    */
    public void setReplaySteps(int replaySteps) {
      this.replaySteps = replaySteps;
    }
  }
}