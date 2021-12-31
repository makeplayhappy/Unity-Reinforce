using UnityEngine;

namespace Reinforce{

    public abstract class Solver {

        public Environment env;
        public DQNOpt opt;

        public Solver(Environment environment, DQNOpt options) {
            env = environment;
            opt = options;
        }

        public DQNOpt getOpt(){
            return opt;
        }

        public Environment getEnv(){
            return env;
        }

        /**
        * Decide an action according to current state
        * @param state current state
        * @returns decided action
        */
        public abstract int decide(float[] stateList);
        public abstract void learn(float r1);
        public abstract void reset();
        public abstract string toJSON();
        public abstract void fromJSON(string json);
    }

}
