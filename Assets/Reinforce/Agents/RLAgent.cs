/*
import { Env, Opt } from 'reinforce-js';

import { SensedObject } from './components/utils/SensedObject';

import { Point2D } from './../../utils/Point2D';

import { World } from './../../World';
import { Agent } from '../Agent';
import { Item } from '../Item';

import { DQNBrain } from './components/DQNBrain';
import { Sensory } from './components/Sensory';
import { WorldObject } from '../WorldObject';
*/
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Reinforce{
  public class RLAgent : MonoBehaviour, IAgent {

    //public readonly sensory: Sensory;
    public DQNBrain brain;

    public int numberOfActions;

    private List<Vector2> actions;

    private float actionAngle = 2f; 
    private float minActionAngle = -8f;
    private float maxActionAngle = 8f;

    private int actionIndex;

    //private consumptionReward: number;
    //private sensoryReward: number;
    private float totalReward;

    private float[] states;
    private int numberStates = 8;



    public bool istraining = true;
    int episodeCount;
    
    float episodeReward;
    float reward;
    public float rotationSpeed = 6f;

    public GameObject ball;
    public Text infoText;
    private Transform ballTransform;
    private Rigidbody ballRb;

    public float timePerDecision = 0.3333f;
    private float nextDecisionTime;

    private Vector3 ballStartPosition;

    private Quaternion wantedRotation;
    private Quaternion startRotation;
    private float lerpAmount = 1.01f;

    public bool isTouching = false;
    public Vector3 positionDelta;

    public Vector2 actionOut = Vector2.zero;




    //private readonly velocityDiscountFactor: number = 0.95;

    /*
    public RLAgent(int id, DQNBrain brain) {

      //super(id, 3, 10, location, new Point2D(0, 0));

      //this.sensory = sensory;
      this.brain = brain;
      this.actionIndex = 0;
      this.reset();

    }

    public RLAgent(int id) {
      generateActions();
      this.brain = createDefaultBrain();
      this.actionIndex = 0;
      this.reset();
    }*/

    void Awake(){
      generateActions();
      states = new float[ numberStates ];
      this.brain = createDefaultBrain();
      this.actionIndex = 0;
      
      
    }

    void Start(){
      ballTransform = ball.transform;
      ballRb = ball.GetComponent<Rigidbody>();
      ballStartPosition = ballTransform.position;

      this.reset();
      
    }

    void FixedUpdate(){
        positionDelta = ballTransform.position - transform.position; 

        if (positionDelta.y < -1.5f || Mathf.Abs(positionDelta.x) > 3.8f || Mathf.Abs(positionDelta.z) > 3.8f){
          reward = -1f;
          learn();
          reset();
          decide();
          episodeCount++;

        }else if( Time.fixedTime > nextDecisionTime && isTouching){
            nextDecisionTime = Time.fixedTime + timePerDecision;
            this.totalReward += getPositionReward();
            learn();
            decide();
            
        }
    }

    void Update(){
      //observe and learn
      if( lerpAmount <= 1f){

            transform.rotation = Quaternion.Lerp(startRotation, wantedRotation, lerpAmount);
            lerpAmount += Time.deltaTime * rotationSpeed;
      }
      //check out of bounds
    }

    //distance from center reward
    private float getPositionReward(){
      reward = 0.01f + Mathf.Clamp(0.1f * (3f - Mathf.Sqrt(positionDelta.x*positionDelta.x+positionDelta.z*positionDelta.z)), 0f , 0.5f);
      return reward;
    }

    //copied from RLAgentFactory
    private DQNBrain createDefaultBrain() {
      //int numberOfStates = this.determineNumberOfStates();
      Environment env = new Environment(0, 0, this.numberStates, this.numberOfActions);

      DQNOpt opt = new DQNOpt();
      opt.setTrainingMode(true); // allows epsilon decay
      opt.setNumberOfHiddenUnits(100); // number of neurons in hidden layer
      opt.setEpsilonDecay(1.0f, 0.1f, 1000000); // initial epsilon for epsilon-greedy policy, 
      opt.setEpsilon(0.05f); // initial epsilon for epsilon-greedy policy, 
      opt.setGamma(0.9f);
      opt.setAlpha(0.005f); // value function learning rate
      opt.setLossClipping(true); // initial epsilon for epsilon-greedy policy, 
      opt.setLossClamp(1.0f); // for robustness
      opt.setRewardClipping(true); // initial epsilon for epsilon-greedy policy, 
      opt.setRewardClamp(1.0f); // initial epsilon for epsilon-greedy policy, 
      opt.setExperienceSize(1000000); //1e6 // size of experience
      opt.setReplayInterval(5); // number of time steps before we add another experience to replay memory
      opt.setReplaySteps(5);
      // outfit brain with environment complexity and specs
      DQNBrain brain = new DQNBrain(env, opt);
      return brain;
    }

/*    //copied from RLAgentFactory
    // number of float observations
    private int determineNumberOfStates() {

      return 10;
    }
*/
    public void reset() {
      this.totalReward = 0f;
      nextDecisionTime = Time.fixedTime + timePerDecision;
      positionDelta = Vector3.zero;

      //this.consumptionReward = 0;
      //this.sensoryReward = 0;
      //this.sensory.reset();

      ballRb.velocity = new Vector3(0f, 0f, 0f);
      ballTransform.position = new Vector3(Random.Range(-1.5f, 1.5f), 0f, Random.Range(-1.5f, 1.5f)) + ballStartPosition;    

    }

    public void load(string brainState) {
      this.brain.load(brainState);
    }

    public void setTrainingModeTo(bool trainingMode) {
      this.brain.setTrainingModeTo(trainingMode);
    }


    public DQNOpt getOpt() {
      return this.brain.getOpt();
    }

    public Environment getEnv() {
      return this.brain.getEnv();
    }

    public void observe(float[] observations) {

      //states = observations;
      /*
      this.sensory.process(world, this);
          *** this does: 
          for (const sensor of this.sensors) {
            sensor.senseObject(world, agent);
          }
          ****
          Senses WorldObjects in its respective range and relative to the agents (sensor owner) location
          ***
          */

    }


    // Collect the observations / sensors
    private void observeStates(){

      states[0] = Mathf.Clamp(transform.rotation.x, -1f, 1f);
      states[1] = Mathf.Clamp(transform.rotation.z, -1f, -1f);

      Vector3 normalisedPositionDelta = positionDelta * 0.3333f; // normalise to bounds is 3 so multiply by 1/3 = 0.3333
      states[2] = Mathf.Clamp(positionDelta.x, -1f, 1f);
      states[3] = Mathf.Clamp(positionDelta.y, -1f, 1f);
      states[4] = Mathf.Clamp(positionDelta.z, -1f, 1f);

      Vector3 normalisedVelo = ballRb.velocity * 0.2f; //assume a max velo of 5

      states[5] = Mathf.Clamp(normalisedVelo.x, -1f, 1f);
      states[6] = Mathf.Clamp(normalisedVelo.y, -1f, 1f);
      states[7] = Mathf.Clamp(normalisedVelo.z, -1f, 1f);
    }

    /**
     * Make a decision based on SensoryState 
     *  brain.decide pushes - float[] state
     */
    public void decide() {

      // this gets the observable data
      //const states = this.sensory.getSensoryState();

      // add proprioception and orientation
      //states.push(this.velocity.x);
      //states.push(this.velocity.y);

      observeStates();

      actionIndex = brain.decide(states);

      //run the new action index
      
      float degreeRotationRange = 8f;
      Vector3 wanterEuler = new Vector3(actions[ actionIndex ][0] , 0, actions[ actionIndex ][1]);
      wantedRotation.eulerAngles = wanterEuler;
      lerpAmount = 0f;
      startRotation = transform.rotation; 

    }

    /**
     * Learning
     */
    public void learn() {
      //this.processSensoryRewards();
      //this.totalReward = this.consumptionReward + this.sensoryReward;
      brain.learn( totalReward );
      totalReward = 0f;
    }


    private void generateActions(){
      actions = new List<Vector2>();
      //balancer has quantised position states -8 .. 8 degrees in X and Z rotations
      for(float x = minActionAngle; x <= maxActionAngle; x = x+actionAngle ){
        for(float z = minActionAngle; z <= maxActionAngle; z = z+actionAngle ){
          actions.Add( new Vector2(x,z) );
        }
      }

      numberOfActions = actions.Count;

    }

    /**
     * Act according to the decision made - this moves the agent in the simulation - Unity will handle this
     * /
    public act(world: World): void {
      this.prepareAction();

      this.determineNextLocation();

      this.onWallCollision(world);
    }

    /**
     * 
     * /
    private prepareAction(): void {
      const speed = 1;
      if (this.actionIndex === 0) {
        this.velocity.x += -speed;
      }
      else if (this.actionIndex === 1) {
        this.velocity.x += speed;
      }
      else if (this.actionIndex === 2) {
        this.velocity.y += -speed;
      }
      else if (this.actionIndex === 3) {
        this.velocity.y += speed;
      }
    }

    private determineNextLocation(): void {
      this.velocity.x *= this.velocityDiscountFactor;
      this.velocity.y *= this.velocityDiscountFactor;
      this.location.x += this.velocity.x;
      this.location.y += this.velocity.y;
    }

    private onWallCollision(world: World): void {
      if (this.location.x < 1) {
        this.location.x = 1;
        this.velocity.x = 0;
        this.velocity.y = 0;
      }
      else if (this.location.x > world.width - 1) {
        this.location.x = world.width - 1;
        this.velocity.x = 0;
        this.velocity.y = 0;
      }
      if (this.location.y < 1) {
        this.location.y = 1;
        this.velocity.x = 0;
        this.velocity.y = 0;
      }
      else if (this.location.y > world.height - 1) {
        this.location.y = world.height - 1;
        this.velocity.x = 0;
        this.velocity.y = 0;
      }
    }
    */

    /**
     * Get rewards from collision and return true if collided.
     * Collisions are interpreted as Consumption.
     * @param Item to be evaluated
     * @returns true if collision happend
     * /
    public processCollision(item: Item): boolean {
      const distance = this.location.getDistanceTo(item.location);
      if (this.isColliding(distance, item)) {
        this.recordCollision(item);
        return true;
      }
      return false;
    }

    private isColliding(distance: number, item: Item) {
      return distance < (this.size + item.size);
    }

    private recordCollision(item: Item): void {
      this.consumptionReward += item.getValue();
      if (item.type === 1) {
        this.sensory.item0CollisionsPerTick++;
      }
      else if (item.type === 2) {
        this.sensory.item1CollisionsPerTick++;
      }
    }
    */

    
    void OnCollisionStay(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" && !isTouching){
            isTouching = true;
        }
    }

    void OnCollisionExit(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" && isTouching ){
            isTouching = false;
        }
    }

    void OnCollisionEnter(Collision collisionInfo) {
        if( collisionInfo.transform.tag == "Player" ){
            isTouching = true;
        }
    }



    /**
     * Sensation-Rewards
     * /
    private processSensoryRewards(): void {
      this.sensoryReward = this.sensory.processRewards(this);
    }
    */
  }
}
