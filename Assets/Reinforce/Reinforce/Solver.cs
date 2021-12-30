using UnityEngine;

namespace Reinforce{

    public abstract class Solver {

        protected Environment env;
        protected Options opt;

        Solver(Environment env, Options opt) {
            this.env = env;
            this.opt = opt;
        }

        public Options getOpt(){
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
