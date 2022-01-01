using System.Collections.Generic;
using UnityEngine;
using Recurrent;

namespace Reinforce{
    public class DQNSolver : Solver {
        // Env
        public int numberOfStates;
        public int numberOfActions;

        // Opts
        public int[] numberOfHiddenUnits;

        public readonly float epsilonMax;
        public readonly float epsilonMin;
        public readonly int epsilonDecayPeriod;
        public readonly float epsilon;
        
        public readonly float gamma;
        public readonly float alpha;
        public readonly bool doLossClipping;
        public readonly float lossClamp;
        public readonly bool doRewardClipping;
        public readonly float rewardClamp;
        public readonly int experienceSize;
        public readonly int keepExperienceInterval;
        public readonly int replaySteps;
        
        // Local
        private Net net;
        private Graph previousGraph;
        private SarsaExperience shortTermMemory;// = new SarsaExperience(){ s0: null, a0: null, r0: null, s1: null, a1: null };
        private List<SarsaExperience> longTermMemory;
        private bool isInTrainingMode;
        private int learnTick;
        private int memoryIndexTick;


        DQNSolver(Environment environ, DQNOpt option) : base( environ, option ) {

            shortTermMemory = new SarsaExperience();
            //super(env, opt);
            
            this.numberOfHiddenUnits = option.numberOfHiddenUnits;
            
            this.epsilonMax = option.epsilonMax;
            this.epsilonMin = option.epsilonMin;
            this.epsilonDecayPeriod = option.epsilonDecayPeriod;
            this.epsilon = option.epsilon;
            
            this.experienceSize = option.experienceSize;
            this.gamma = option.gamma;
            this.alpha = option.alpha;
            this.doLossClipping = option.doLossClipping;
            this.lossClamp = option.lossClamp;
            this.doRewardClipping = option.doRewardClipping;
            this.rewardClamp = option.rewardClamp;
            
            this.keepExperienceInterval = option.keepExperienceInterval;
            this.replaySteps = option.replaySteps;
            
            this.isInTrainingMode = option.trainingMode;
            
            this.reset();
        }

        public override void reset() {
            this.numberOfHiddenUnits = this.opt.numberOfHiddenUnits;
            this.numberOfStates = this.env.numberOfStates;
            this.numberOfActions = this.env.numberOfActions;

            NetOpts netOpts = new NetOpts(this.numberOfStates,this.numberOfHiddenUnits,this.numberOfActions);

            this.net = new Net(netOpts);

            this.learnTick = 0;
            this.memoryIndexTick = 0;
            
            this.shortTermMemory.s0 = null;
            this.shortTermMemory.a0 = null;
            this.shortTermMemory.r0 = null;
            this.shortTermMemory.s1 = null;
            this.shortTermMemory.a1 = null;

            this.longTermMemory = new List<SarsaExperience>(); // could be new SarsaExperience[experienceSize]()
        }

        /**
        * Sets the training mode of the agent to the given state.
        * given: true => training mode on, linearly decaying epsilon 
        * given: false => deployment mode on, constant epsilon
        * (Epsilon is the threshold for controlling the epsilon greedy policy (or exploration via random actions))
        * @param trainingMode true if training mode should be switched on, false if training mode should be switched off
        */
        public void setTrainingModeTo(bool trainingMode) {
            this.isInTrainingMode = trainingMode;
        }

        /**
        * Returns the current state of training mode
        * @returns true if is in training mode, else false.
        */
        public bool getTrainingMode(){
            return this.isInTrainingMode;
        }

        /**
        * Transforms Agent to (ready-to-stringify) JSON object
        */
        public override string toJSON(){
            /*
            const j = {
            ns: this.numberOfStates,
            nh: this.numberOfHiddenUnits,
            na: this.numberOfActions,
            net: Net.toJSON(this.net)
            };
            return j;*/
            return "";
        }

        /**
        * Loads an Agent from a (already parsed) JSON object
        * @param json with properties `nh`, `ns`, `na` and `net`
        */
        public override void fromJSON(string json) { //: { ns, nh, na, net }
            /*
            this.numberOfStates = json.ns;
            this.numberOfHiddenUnits = json.nh;
            this.numberOfActions = json.na;
            this.net = Net.fromJSON(json.net);
            */
        }

        /**
        * Decide an action according to current state
        * @param state current state
        * @returns index of argmax action
        */
        public override int decide(float[] state) {
            Mat stateVector = new Mat(this.numberOfStates, 1);

            stateVector.setFrom(state);

            int actionIndex = this.epsilonGreedyActionPolicy(stateVector);

            this.shiftStateMemory(stateVector, actionIndex);

            return actionIndex;
        }

        protected int epsilonGreedyActionPolicy(Mat stateVector) {
            int actionIndex = 0;

            if (Utils.rand() < this.currentEpsilon()) { // greedy Policy Filter
                actionIndex = Utils.randi(0, this.numberOfActions);
            } else {
            // Q function
                Mat actionVector = this.forwardQ(stateVector);
                actionIndex = Utils.argmax(actionVector.w); // returns index of argmax action 
            }
            return actionIndex;
        }

        /**
        * Determines the current epsilon.
        */
        protected float currentEpsilon() {
            if (this.isInTrainingMode) {
                if (this.learnTick < this.epsilonDecayPeriod) {
                    return this.epsilonMax - (this.epsilonMax - this.epsilonMin) / this.epsilonDecayPeriod * this.learnTick;
                } else {
                    return this.epsilonMin;
                }
            } else {
                return this.epsilon;
            }
        }

        /**
        * Determine Outputs based on Forward Pass
        * @param stateVector Matrix with states
        * @return Matrix (Vector) with predicted actions values
        */
        protected Mat forwardQ(Mat? stateVector) {
            Graph graph = new Graph();  // without backprop option
            Mat a2Mat = this.determineActionVector(graph, stateVector);
            return a2Mat;
        }

        /**
        * Determine Outputs based on Forward Pass
        * @param stateVector Matrix with states
        * @return Matrix (Vector) with predicted actions values
        */
        protected Mat backwardQ(Mat? stateVector) {
            Graph graph = new Graph();
            graph.memorizeOperationSequence(true);  // with backprop option
            Mat a2Mat = this.determineActionVector(graph, stateVector);
            return a2Mat;
        }


        protected Mat determineActionVector(Graph graph, Mat stateVector ) {
            Mat a2mat = this.net.forward(stateVector, graph);
            this.backupGraph(graph); // back this up
            return a2mat;
        }

        // - TODO - this shouldn't pass by reference.... check this
        protected void backupGraph(Graph graph) {
            this.previousGraph = graph;
        }

        protected void shiftStateMemory(Mat stateVector, int actionIndex) {
            this.shortTermMemory.s0 = this.shortTermMemory.s1;
            this.shortTermMemory.a0 = this.shortTermMemory.a1;
            this.shortTermMemory.s1 = stateVector;
            this.shortTermMemory.a1 = actionIndex;
        }

        /**
        * perform an update on Q function
        * @param r current reward passed to learn
        */
        public override void learn(float r) {
            if (this.shortTermMemory.r0 != null && this.alpha > 0) {
                this.learnFromSarsaTuple(this.shortTermMemory);
                this.addToReplayMemory();
                this.limitedSampledReplayLearning();
            }
            this.shiftRewardIntoMemory(r);
        }

        private void shiftRewardIntoMemory(float r) {
            this.shortTermMemory.r0 = this.clipReward(r);
        }

        /**
        * Clips Reward, If doRewardClipping is activated
        * @param r current reward
        */
        protected float clipReward(float r) {
            return this.doRewardClipping ? Mathf.Sign(r) * Mathf.Min(Mathf.Abs(r), this.rewardClamp) : r;
        }

        /**
        * Learn from sarsa tuple
        * @param {SarsaExperience} sarsa Object containing states, actions and reward of t & t-1
        */
        protected void learnFromSarsaTuple(SarsaExperience sarsa ) {
            float q1Max = this.getTargetQ(sarsa.s1, sarsa.r0);
            Mat q0ActionVector = this.backwardQ(sarsa.s0);
            //float q0Max = 0f;
            //if( sarsa.a0 != null){
                float q0Max = q0ActionVector.w[(int)sarsa.a0];
            //}

            // Loss_i(w_i) = [(r0 + gamma * Q'(s',a') - Q(s,a)) ^ 2]
            float loss = q0Max - q1Max;
            loss = this.clipLoss(loss);

            //if( sarsa.a0 != null){
                q0ActionVector.dw[(int)sarsa.a0] = loss;
            //}
            this.previousGraph.backward();

            // discount all weights of net depending on their gradients
            this.net.update(this.alpha);
        }

        protected float getTargetQ(Mat s1 , float? r0) {
            //if(r0 == null){
            //    r0 = 0f;
            //}
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')
            Mat targetActionVector = this.forwardQ(s1);
            int targetActionIndex = Utils.argmax(targetActionVector.w);
            float qMax = (float)r0 + this.gamma * targetActionVector.w[targetActionIndex];
            return qMax;
        }

        /**
        * Clip loss to interval of [-delta, delta], e.g. [-1, 1] (Derivative Huber Loss)
        * @returns {number} limited tdError
        */
        protected float clipLoss(float loss) {
            if (this.doLossClipping) {
            if (loss > this.lossClamp) {
                loss = this.lossClamp;
            }
            else if (loss < -this.lossClamp) {
                loss = -this.lossClamp;
            }
            }
            return loss;
        }

        protected void addToReplayMemory() {
            if (this.learnTick % this.keepExperienceInterval == 0) {
                this.addShortTermToLongTermMemory();
            }
            this.learnTick++;
        }

        protected void addShortTermToLongTermMemory() {
            SarsaExperience sarsa = this.extractSarsaExperience();
            this.longTermMemory[this.memoryIndexTick] = sarsa;
            this.memoryIndexTick++;
            if (this.memoryIndexTick > this.experienceSize - 1) { // roll over
                this.memoryIndexTick = 0;
            }
        }

        protected SarsaExperience extractSarsaExperience() {

            Mat s0 = new Mat(this.shortTermMemory.s0.rows, this.shortTermMemory.s0.cols);
            s0.setFrom(this.shortTermMemory.s0.w);
            Mat s1 = new Mat(this.shortTermMemory.s1.rows, this.shortTermMemory.s1.cols);
            s1.setFrom(this.shortTermMemory.s1.w);

            SarsaExperience sarsa = new SarsaExperience(
                s0,
                this.shortTermMemory.a0,
                this.shortTermMemory.r0,
                s1,
                this.shortTermMemory.a1);

            return sarsa;
        }

        /**
        * Sample some additional experience (minibatches) from replay memory and learn from it
        */
        protected void limitedSampledReplayLearning() {
            for (int i = 0; i < this.replaySteps; i++) {
                int ri = Utils.randi(0, this.longTermMemory.Count); // todo: priority sweeps?
                SarsaExperience sarsa = this.longTermMemory[ri];
                this.learnFromSarsaTuple(sarsa);
            }
        }
        
    }
}
