//import { DQNSolver, Env, DQNOpt } from 'reinforce-js';


namespace Reinforce{
  public class DQNBrain : DQNSolver {

    
    public DQNBrain(Environment environ, DQNOpt option) : base( environ, option ) {
      
    }

    /**
    * Load brain State into current solver
    * @param brainState - as JSON string
    */
    public void load(string brainStateJson) {
      this.fromJSON(brainStateJson); //  as { ns, nh, na, net }
    }
  }

}
