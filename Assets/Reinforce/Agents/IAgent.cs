/*import { World } from './../World';
import { Item } from './Item';
import { Opt, Env } from 'reinforce-js';*/
namespace Reinforce{

  interface IAgent {
    
    void reset();
    void load(string brainStateJson) ;

    void observe(float[] observations); //CollectObservations
    void decide();
    //void act(World world);
    void learn(float reward);
    /**
     * Get rewards from collision and return true if collided.
     * @param Item to be evaluated
     * @returns true if item was collided
     */
    //bool processCollision(Item item);
    DQNOpt getOpt();
    Environment getEnv();
  }
}
