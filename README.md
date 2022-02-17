# Unity-Reinforce


### History
Crosspile from Typescript to C# of a small subset of [Menno van Rahden's](https://github.com/mvrahden) Reinforcement Learning libraries [reinforce-js](https://github.com/mvrahden/reinforce-js/) and [recurrent-js](https://github.com/mvrahden/recurrent-js/) to be usable in C Sharp and Unity3D. Kudos to him for writing such an elegant library in TypeScript, some of the prettiest code I've stumbled upon! These are reachitecturings of [Andrej Karpathy's](https://github.com/karpathy/) excellent [reinforcejs](https://github.com/karpathy/reinforcejs) and [recurrentjs](https://github.com/karpathy/recurrentjs).  

The graph operation stack has been modified into a class as Unity only supports up to .Net v4, and I can't find a matching way to store the back propogation stack in a similar way as is done in the JS code. 
  

Specifically this repository is to get the DQN solver from these libs working in Unity3D
 

Maybe working sometime in 2022, don't hold your breath!
