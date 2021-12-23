using UnityEngine;

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
        protected Net net;
        protected Graph previousGraph;
        protected SarsaExperience shortTermMemory;// = new SarsaExperience(){ s0: null, a0: null, r0: null, s1: null, a1: null };
        protected SarsaExperience[] longTermMemory;
        protected bool isInTrainingMode;
        protected int learnTick;
        protected int memoryIndexTick;


        public DQNSolver(Environment env, Options DQNOpt) : base( env, DQNOpt ) {
            //super(env, opt);
            this.numberOfHiddenUnits = opt.get('numberOfHiddenUnits');
            
            this.epsilonMax = opt.get('epsilonMax');
            this.epsilonMin = opt.get('epsilonMin');
            this.epsilonDecayPeriod = opt.get('epsilonDecayPeriod');
            this.epsilon = opt.get('epsilon');
            
            this.experienceSize = opt.get('experienceSize');
            this.gamma = opt.get('gamma');
            this.alpha = opt.get('alpha');
            this.doLossClipping = opt.get('doLossClipping');
            this.lossClamp = opt.get('lossClamp');
            this.doRewardClipping = opt.get('doRewardClipping');
            this.rewardClamp = opt.get('rewardClamp');
            
            this.keepExperienceInterval = opt.get('keepExperienceInterval');
            this.replaySteps = opt.get('replaySteps');
            
            this.isInTrainingMode = opt.get('trainingMode');

            this.reset();
        }

        public void reset() {
            this.numberOfHiddenUnits = this.opt.get('numberOfHiddenUnits');
            this.numberOfStates = this.env.get('numberOfStates');
            this.numberOfActions = this.env.get('numberOfActions');

            const netOpts: NetOpts = {
            architecture: {
                inputSize: this.numberOfStates,
                hiddenUnits: this.numberOfHiddenUnits,
                outputSize: this.numberOfActions
            }
            };
            this.net = new Net(netOpts);

            this.learnTick = 0;
            this.memoryIndexTick = 0;
            
            this.shortTermMemory.s0 = null;
            this.shortTermMemory.a0 = null;
            this.shortTermMemory.r0 = null;
            this.shortTermMemory.s1 = null;
            this.shortTermMemory.a1 = null;

            this.longTermMemory = [];
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
        public string toJSON(){
            const j = {
            ns: this.numberOfStates,
            nh: this.numberOfHiddenUnits,
            na: this.numberOfActions,
            net: Net.toJSON(this.net)
            };
            return j;
        }

        /**
        * Loads an Agent from a (already parsed) JSON object
        * @param json with properties `nh`, `ns`, `na` and `net`
        */
        public void fromJSON(string json: { ns, nh, na, net }) {
            this.numberOfStates = json.ns;
            this.numberOfHiddenUnits = json.nh;
            this.numberOfActions = json.na;
            this.net = Net.fromJSON(json.net);
        }

        /**
        * Decide an action according to current state
        * @param state current state
        * @returns index of argmax action
        */
        public int decide(float[] state) {
            Mat stateVector = new Mat(this.numberOfStates, 1);

            stateVector.setFrom(state);

            int actionIndex = this.epsilonGreedyActionPolicy(stateVector);

            this.shiftStateMemory(stateVector, actionIndex);

            return actionIndex;
        }

        protected int epsilonGreedyActionPolicy(Mat stateVector) {
            int actionIndex = 0;

            if (Math.random() < this.currentEpsilon()) { // greedy Policy Filter
                actionIndex = Utils.randi(0, this.numberOfActions);
            } else {
            // Q function
                const actionVector = this.forwardQ(stateVector);
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
        protected Mat forwardQ(stateVector: Mat | null) {
            const graph = new Graph();  // without backprop option
            const a2Mat = this.determineActionVector(graph, stateVector);
            return a2Mat;
        }

        /**
        * Determine Outputs based on Forward Pass
        * @param stateVector Matrix with states
        * @return Matrix (Vector) with predicted actions values
        */
        protected Mat backwardQ(stateVector: Mat | null) {
            const graph = new Graph();
            graph.memorizeOperationSequence(true);  // with backprop option
            const a2Mat = this.determineActionVector(graph, stateVector);
            return a2Mat;
        }


        protected Mat determineActionVector(graph: Graph, stateVector: Mat) {
            const a2mat = this.net.forward(stateVector, graph);
            this.backupGraph(graph); // back this up
            return a2mat;
        }

        protected void backupGraph(graph: Graph) {
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
        public void learn(float r) {
            if (this.shortTermMemory.r0 && this.alpha > 0) {
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
            return this.doRewardClipping ? Math.sign(r) * Math.min(Math.abs(r), this.rewardClamp) : r;
        }

        /**
        * Learn from sarsa tuple
        * @param {SarsaExperience} sarsa Object containing states, actions and reward of t & t-1
        */
        protected void learnFromSarsaTuple(SarsaExperience sarsa ) {
            const q1Max = this.getTargetQ(sarsa.s1, sarsa.r0);
            const q0ActionVector = this.backwardQ(sarsa.s0);
            const q0Max = q0ActionVector.w[sarsa.a0];

            // Loss_i(w_i) = [(r0 + gamma * Q'(s',a') - Q(s,a)) ^ 2]
            let loss = q0Max - q1Max;
            loss = this.clipLoss(loss);

            q0ActionVector.dw[sarsa.a0] = loss;
            this.previousGraph.backward();

            // discount all weights of net depending on their gradients
            this.net.update(this.alpha);
        }

        protected float getTargetQ(Mat s1: , float r0) {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')
            const targetActionVector = this.forwardQ(s1);
            const targetActionIndex = Utils.argmax(targetActionVector.w);
            const qMax = r0 + this.gamma * targetActionVector.w[targetActionIndex];
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
            if (this.learnTick % this.keepExperienceInterval === 0) {
                this.addShortTermToLongTermMemory();
            }
            this.learnTick++;
        }

        protected void addShortTermToLongTermMemory() {
            const sarsa = this.extractSarsaExperience();
            this.longTermMemory[this.memoryIndexTick] = sarsa;
            this.memoryIndexTick++;
            if (this.memoryIndexTick > this.experienceSize - 1) { // roll over
                this.memoryIndexTick = 0;
            }
        }

        protected SarsaExperience extractSarsaExperience() {
            const s0 = new Mat(this.shortTermMemory.s0.rows, this.shortTermMemory.s0.cols);
            s0.setFrom(this.shortTermMemory.s0.w);
            const s1 = new Mat(this.shortTermMemory.s1.rows, this.shortTermMemory.s1.cols);
            s1.setFrom(this.shortTermMemory.s1.w);
            const sarsa = {
                s0,
                a0: this.shortTermMemory.a0,
                r0: this.shortTermMemory.r0,
                s1,
                a1: this.shortTermMemory.a1
            };
            return sarsa;
        }

        /**
        * Sample some additional experience (minibatches) from replay memory and learn from it
        */
        protected void limitedSampledReplayLearning() {
            for (let i = 0; i < this.replaySteps; i++) {
            const ri = Utils.randi(0, this.longTermMemory.length); // todo: priority sweeps?
            const sarsa = this.longTermMemory[ri];
            this.learnFromSarsaTuple(sarsa);
            }
        }
        
    }
}
